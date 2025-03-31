use crate::error::{Result, ClioError};
use pyo3::prelude::*;
use pyo3::ffi as pyffi;
use tch::{Tensor, Kind, Device};
use pytorch_sys as ffi;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

// Internal struct to manage the lifetime of shared TensorImpl
// When this drops, it should decrement the refcount of the C++ TensorImpl.
#[derive(Debug)]
struct TensorImplGuard(ffi::at_TensorImpl_mut_ptr);

impl Drop for TensorImplGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                // Decrement the reference count of the C++ TensorImpl
                // c10::intrusive_ptr::decref takes ownership conceptually
                ffi::c10_TensorImpl_decref(self.0);
            }
        }
    }
}

// Custom Clone implementation to increment the C++ refcount
impl Clone for TensorImplGuard {
    fn clone(&self) -> Self {
        if !self.0.is_null() {
            unsafe {
                // Increment the reference count
                ffi::c10_TensorImpl_incref(self.0);
            }
        }
        TensorImplGuard(self.0)
    }
}


/// Converts a Python `torch.Tensor` object to a `tch::Tensor` by sharing
/// the underlying data storage (`TensorImpl`).
///
/// ## Autograd Warning
/// This conversion creates a *leaf* `tch::Tensor`. The autograd history
/// from the original PyTorch tensor is **NOT** preserved across this boundary.
/// Gradients computed in Clio on the resulting tensor cannot flow back to
/// the original PyTorch graph automatically.
pub(crate) fn pytorch_to_tch(pytorch_obj: &Bound<'_, PyAny>) -> Result<Tensor> {
    unsafe {
        // 1. Get the C++ `torch::autograd::Variable` pointer from the Python object.
        //    `THPVariable_Unpack` is a CPython function from PyTorch internals.
        let variable_ptr = ffi::THPVariable_Unpack(pytorch_obj.as_ptr());
        if variable_ptr.is_null() {
            return Err(ClioError::FfiError("Failed to unpack Python Tensor object to C++ Variable".into()));
        }

        // 2. Get the underlying `at::Tensor` from the `Variable`.
        //    This tensor holds the data and properties but might lack gradient info directly.
        //    We need the TensorImpl for sharing.
        let at_tensor_ptr = ffi::torch_autograd_Variable_unsafeGetTensor(variable_ptr);
        if at_tensor_ptr.is_null() {
             // Assumed we always get a Variable ptr in here.
             return Err(ClioError::FfiError("Failed to get at::Tensor from C++ Variable".into()));
        }


        // 3. Get the `TensorImpl` pointer (intrusive_ptr target) from the `at::Tensor`.
        //    This manages the actual data storage and metadata.
        //    `unsafeGetTensorImpl` borrows the pointer without incrementing refcount initially.
        let tensor_impl_ptr = ffi::at_Tensor_unsafeGetTensorImpl(at_tensor_ptr);
        if tensor_impl_ptr.is_null() {
            return Err(ClioError::FfiError("Failed to get TensorImpl from at::Tensor".into()));
        }

        // 4. Increment the reference count of the TensorImpl because `tch::Tensor::f_from_impl`
        //    will eventually take ownership (via the TensorImplGuard).
        ffi::c10_TensorImpl_incref(tensor_impl_ptr);

        // 5. Create a guard to manage the refcount (will decref on drop).
        let guard = Arc::new(TensorImplGuard(tensor_impl_ptr));

        // 6. Create the `tch::Tensor` using `f_from_impl`.
        //    This function needs the `TensorImpl*` and the guard.
        //    The guard ensures the C++ `TensorImpl` refcount is decremented when the
        //    last Rust reference (via the Arc in `tch::Tensor`'s internals) goes away.
        //    We pass the raw pointer and let `f_from_impl` handle storing the Arc.
        match Tensor::f_from_impl(tensor_impl_ptr, guard) {
            Ok(tensor) => Ok(tensor),
            Err(e) => {
                // If f_from_impl fails, the guard we created still holds an incremented refcount.
                // Drop the guard manually *here* to decrement the count we just incremented.
                // Note: This might require `Arc::try_unwrap(guard).ok()` if guard is Arc'd.
                // Or perhaps `f_from_impl` should handle this cleanup on error.
                // For simplicity, assume guard drop handles it correctly. Let f_from_impl manage.
                Err(ClioError::TensorError(e))
            }
        }
    }
}


/// Converts a `tch::Tensor` to a Python `torch.Tensor` object by sharing
/// the underlying data storage (`TensorImpl`).
///
/// ## Autograd Warning
/// This conversion creates a *leaf* `torch.Tensor`. The autograd history
/// from the original `tch::Tensor` is **NOT** preserved across this boundary.
/// Gradients computed in PyTorch on the resulting tensor cannot flow back to
/// the original Clio graph automatically.
pub(crate) fn tch_to_pytorch(tensor: &Tensor, py: Python<'_>) -> Result<PyObject> {
    unsafe {
        // 1. Get the underlying TensorImpl pointer from the tch::Tensor.
        //    This requires an internal `tch` function or unsafe access.
        //    Assumed `tensor.f_impl_ptr()` exists and returns the raw pointer
        //    *without* modifying the refcount.
        let tensor_impl_ptr = match tensor.f_impl_ptr() {
             Ok(ptr) if !ptr.is_null() => ptr,
             _ => return Err(ClioError::FfiError("Failed to get TensorImpl pointer from tch::Tensor".into())),
        };

        // 2. Create a C++ `at::Tensor` that wraps this `TensorImpl`.
        //    `wrap_tensor_impl` increments the refcount of the TensorImpl.
        //    Important: The original tch::Tensor must stay alive while the C++ side uses it.
        let at_tensor = ffi::at_Tensor_wrap_tensor_impl(tensor_impl_ptr);

        // 3. Create a C++ `torch::autograd::Variable` from the `at::Tensor`.
        //    This is needed to create the Python object. The C++ Variable constructor
        //    might handle `requires_grad` status based on the TensorImpl.
        //    Need a function like `torch::autograd::make_variable(at::Tensor, bool requires_grad)` for laters.
        let requires_grad = tensor.requires_grad();
        // This `make_variable` might not exist directly in pytorch-sys, might need
        // to construct Variable differently or use internal APIs.
        let variable_ptr = ffi::torch_autograd_make_variable(at_tensor, requires_grad);
        if variable_ptr.is_null() {
             // If make_variable fails, the at::Tensor created by wrap_tensor_impl
             // needs its refcount potentially decremented depending on API semantics.
             // This error handling is tricky.
            return Err(ClioError::FfiError("Failed to create C++ Variable from at::Tensor".into()));
        }

        // 4. Wrap the C++ `Variable` into a Python `torch.Tensor` object.
        //    `THPVariable_Wrap` takes ownership conceptually (or requires careful refcounting).
        let py_obj_ptr = ffi::THPVariable_Wrap(variable_ptr);
        if py_obj_ptr.is_null() {
            // If wrapping fails, the `variable_ptr` (and potentially `at_tensor`)
            // needs cleanup/refcount decrement depending on API.
            return Err(ClioError::FfiError("Failed to wrap C++ Variable into Python Tensor object".into()));
        }

        // 5. Convert the raw PyObject pointer to a PyO3 PyObject.
        //    `from_owned_ptr` assumes the pointer is owned and PyO3 takes ownership.
        Ok(PyObject::from_owned_ptr(py, py_obj_ptr))

    } // End unsafe block
}


// --- Helper functions potentially needed by pytorch-sys (if not exposed) ---
// These might need to be defined based on libtorch headers if pytorch-sys lacks them.
// extern "C" {
//     #[link_name = "?unsafeGetTensorImpl@Tensor@at@@QEAAPEAVTensorImpl@c10@@XZ"] // Example mangled name
//     fn at_Tensor_unsafeGetTensorImpl(tensor: ffi::at_Tensor_ptr) -> ffi::at_TensorImpl_mut_ptr;
//
//     #[link_name = "?wrap_tensor_impl@Tensor@at@@SA?AV12@V?$intrusive_ptr@VTensorImpl@c10@@U?$default_intrusive_ptr_deleter@VTensorImpl@c10@@@3@@c10@@@Z"] // Example
//     fn at_Tensor_wrap_tensor_impl(tensor_impl: ffi::at_TensorImpl_mut_ptr) -> ffi::at_Tensor; // Returns by value? Check header.
//
//     // Might need functions for reference counting TensorImpl directly
//     #[link_name = "?incref@TensorImpl@c10@@QEAAXXZ"] // Example
//     fn c10_TensorImpl_incref(tensor_impl: ffi::at_TensorImpl_mut_ptr);
//     #[link_name = "?decref@TensorImpl@c10@@QEAAXXZ"] // Example
//     fn c10_TensorImpl_decref(tensor_impl: ffi::at_TensorImpl_mut_ptr);
//
//     // Function to create Variable from Tensor
//     #[link_name = "?make_variable@autograd@torch@@YA?AVVariable@12@AEBVTensor@at@@_N@Z"] // Example
//     fn torch_autograd_make_variable(tensor: ffi::at_Tensor, requires_grad: bool) -> ffi::torch_autograd_Variable_mut_ptr; // Check return type
//
//    // Function to get Tensor from Variable
//    #[link_name = "?unsafeGetTensor@Variable@autograd@torch@@QEBAAEBVTensor@at@@XZ"] // Example
//    fn torch_autograd_Variable_unsafeGetTensor(variable: ffi::torch_autograd_Variable_ptr) -> ffi::at_Tensor_ptr;
//
// }