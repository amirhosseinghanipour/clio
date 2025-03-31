use crate::error::{Result, ClioError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::{Arc, Mutex};
use tch::{Tensor, Kind, Device};

mod ffi;

#[pyclass(name = "Tensor")]
#[derive(Debug, Clone)]
pub struct ClioTensor {
    // Wrap the tch::Tensor in Arc<Mutex> for thread safety & PyO3 interior mutability
    pub(crate) inner: Arc<Mutex<Tensor>>,
}

// Internal constructor (not exposed to Python)
impl ClioTensor {
    pub(crate) fn new(tensor: Tensor) -> Self {
        ClioTensor { inner: Arc::new(Mutex::new(tensor)) }
    }

    // Helper to lock and access inner tensor
    #[inline]
    fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, Tensor>> {
        self.inner.lock()
    }
}

#[pymethods]
impl ClioTensor {
    #[classmethod]
    #[pyo3(signature = (pytorch_tensor))]
    fn from_pytorch(_cls: &Bound<'_, PyType>, pytorch_tensor: &Bound<'_, PyAny>) -> Result<Self> {
        // Use the FFI bridge
        let tensor = ffi::pytorch_to_tch(pytorch_tensor)?;
        Ok(Self::new(tensor))
    }

    #[classmethod]
    #[pyo3(signature = (numpy_array))]
    fn from_numpy(_cls: &Bound<'_, PyType>, numpy_array: &Bound<'_, PyAny>) -> Result<Self> {
        // Convert numpy to PyTorch first (requires Python call), then use FFI
        let py = numpy_array.py();
        let torch = py.import_bound("torch")?;
        let torch_tensor = torch.call_method1("from_numpy", (numpy_array,))?;
        Self::from_pytorch(_cls, &torch_tensor)
    }

    fn to_pytorch(&self) -> Result<PyObject> {
         let py = Python::acquire_gil();
         // Use the FFI bridge
         ffi::tch_to_pytorch(&self.lock()?, py.python())
    }

    fn to_numpy(&self) -> Result<PyObject> {
         let py_tensor_obj = self.to_pytorch()?;
         Python::with_gil(|py| {
             let numpy_array = py_tensor_obj.bind(py).call_method0("numpy")?;
             Ok(numpy_array.to_object(py))
         })
    }

    // --- Properties ---
    #[getter]
    fn shape(&self) -> Result<Vec<i64>> {
        Ok(self.lock()?.size())
    }

    #[getter]
    fn ndim(&self) -> Result<usize> {
        Ok(self.lock()?.size().len())
    }

    #[getter]
    fn dtype(&self) -> Result<String> {
        let kind = self.lock()?.kind();
        Ok(format!("{:?}", kind).to_lowercase())
    }

    #[getter]
    fn device(&self) -> Result<String> {
        let device = self.lock()?.device();
        Ok(crate::device::device_to_string(device))
    }

    #[getter]
    fn requires_grad(&self) -> Result<bool> {
        Ok(self.lock()?.requires_grad())
    }

    #[setter]
    fn set_requires_grad(&self, value: bool) -> Result<()> {
        // Note: set_requires_grad returns a *new* tensor in tch-rs
        // We need to update the tensor *inside* the mutex
        let mut guard = self.lock()?;
        *guard = guard.set_requires_grad(value);
        Ok(())
    }

    #[getter]
    fn grad(&self) -> Result<Option<ClioTensor>> {
        let grad_opt = self.lock()?.grad();
        Ok(grad_opt.map(ClioTensor::new))
    }

    // --- Autograd ---
    fn backward(&self) -> Result<()> {
        self.lock()?.backward();
        Ok(())
    }

    // --- Data Access ---
    /// Returns the single scalar value of a 0-dim tensor.
    /// Raises an error if the tensor is not 0-dim or not on the CPU.
    fn item(&self) -> Result<PyObject> {
        let tensor = self.lock()?;
        if !tensor.size().is_empty() {
            return Err(ClioError::TypeError("Only 0-dim tensors can be converted to Python scalars with .item()".into()));
        }
        // Ensure tensor is on CPU for safe extraction
        let cpu_tensor = tensor.to_device(Device::Cpu);
        let py = Python::acquire_gil().python();
        match cpu_tensor.kind() {
            Kind::Float => Ok(f32::from(&cpu_tensor).to_object(py)),
            Kind::Double => Ok(f64::from(&cpu_tensor).to_object(py)),
            Kind::Int => Ok(i32::from(&cpu_tensor).to_object(py)),
            Kind::Int64 => Ok(i64::from(&cpu_tensor).to_object(py)),
            Kind::Bool => Ok(bool::from(&cpu_tensor).to_object(py)),
            other => Err(ClioError::TypeError(format!("Cannot convert dtype {:?} to Python scalar using .item()", other))),
        }
    }


    // --- Basic Operations ---
    fn __add__(&self, other: &Bound<'_, PyAny>) -> Result<ClioTensor> {
         let other_tensor = py_to_clio_tensor(other)?; // Use helper
         let locked_self = self.lock()?;
         let locked_other = other_tensor.lock()?;
         let result = locked_self.f_add(&*locked_other)?;
         Ok(ClioTensor::new(result))
    }

     fn __radd__(&self, other: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        self.__add__(other) // Addition is commutative
    }

    // TODO: Implement __sub__, __mul__, __matmul__, __getitem__, etc.

    // --- Representation ---
    fn __str__(&self) -> String {
        // Provide a concise representation, avoid printing large tensors
        let guard = self.lock().unwrap(); // Use unwrap in display methods
        let shape = guard.size();
        let device = guard.device();
        let kind = guard.kind();
        let grad_str = if guard.requires_grad() { ", grad" } else { "" };
        format!("clio.Tensor(shape={:?}, device='{}', dtype='{:?}'{})",
            shape, crate::device::device_to_string(device), kind, grad_str)
    }

    fn __repr__(&self) -> String {
         // More detailed repr, potentially showing some data for small tensors
         let guard = self.lock().unwrap();
         // Limit printing data for large tensors
         let data_str = if guard.numel() < 10 {
             format!("\ndata={}", *guard) // Use tch's Display impl
         } else {
             "".to_string() // Avoid printing large data
         };
         format!("{}{}", self.__str__(), data_str).replace("\n", "\n ") // Indent data part
    }

    fn __len__(&self) -> Result<usize> {
        let size = self.lock()?.size();
        if size.is_empty() {
            Err(ClioError::TypeError("len() of a 0-d tensor".into()))
        } else {
            Ok(size[0] as usize)
        }
    }
}

// Helper to convert PyAny (expecting ClioTensor, PyTorch Tensor, NumPy array, number) to ClioTensor
// Takes Bound types now for better PyO3 integration.
pub(crate) fn py_to_clio_tensor(obj: &Bound<'_, PyAny>) -> Result<ClioTensor> {
    let py = obj.py();
    // Check if it's already a ClioTensor instance
    if let Ok(clio_tensor) = obj.extract::<PyRef<ClioTensor>>() {
        Ok(clio_tensor.clone())
    }
    // Check if it's a PyTorch tensor using isinstance (more robust than type name)
    else if let Ok(torch) = py.import_bound("torch") {
        let tensor_type = torch.getattr("Tensor")?;
         if obj.is_instance(&tensor_type)? {
             // Use FFI bridge
            let tensor = ffi::pytorch_to_tch(obj)?;
            return Ok(ClioTensor::new(tensor));
         }
    }
    // Check if it's a NumPy array using isinstance
    else if let Ok(np) = py.import_bound("numpy") {
         let ndarray_type = np.getattr("ndarray")?;
         if obj.is_instance(&ndarray_type)? {
            return ClioTensor::from_numpy(obj.py().get_type_bound::<ClioTensor>(), obj);
         }
    }
    // Handle Python numbers
    else if let Ok(val) = obj.extract::<f64>() {
        Ok(ClioTensor::new(Tensor::from(val).to_kind(Kind::Double))) // Use Double for f64
    } else if let Ok(val) = obj.extract::<f32>() {
         Ok(ClioTensor::new(Tensor::from(val).to_kind(Kind::Float)))
    } else if let Ok(val) = obj.extract::<i64>() {
        Ok(ClioTensor::new(Tensor::from(val).to_kind(Kind::Int64)))
    } else if let Ok(val) = obj.extract::<bool>() {
         Ok(ClioTensor::new(Tensor::from(if val {1u8} else {0u8}).to_kind(Kind::Bool)))
    }
    // TODO: Handle lists/tuples by converting to tensor
    else {
         let type_name = obj.get_type().qualname()?;
         Err(ClioError::TypeError(format!(
             "Cannot convert Python type '{}' to clio.Tensor", type_name
         )))
    }
}