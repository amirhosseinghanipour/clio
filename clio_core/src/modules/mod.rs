use crate::error::Result;
use crate::tensor::ClioTensor;
use pyo3::prelude::*;
use tch::{Device, nn::VarStore};
use std::sync::{Arc, RwLock};
use std::fmt::Debug;

// Re-export PyClasses from submodules
pub use self::linear::PyLinear;
pub use self::activations::PyReLU;
pub use self::sequential::PySequential;

pub mod linear;
pub mod activations;
pub mod sequential;


/// Common trait for neural network modules (layers, models).
/// This trait operates purely on the Rust side using ClioTensor.
pub trait Module: Debug + Send + Sync {
    /// Performs the forward pass.
    fn forward(&self, xs: &ClioTensor) -> Result<ClioTensor>;

    /// Returns a reference to the module's VarStore (if applicable).
    /// Layers that don't own parameters might return None or an empty VarStore.
    fn varstore(&self) -> Option<Arc<RwLock<VarStore>>>;

    /// Moves the module's parameters (within its VarStore) to the specified device.
    fn to(&self, device: Device) -> Result<()>;

    /// Zeros the gradients of all parameters in the module's VarStore.
    fn zero_grad(&self) -> Result<()> {
        if let Some(vs_arc) = self.varstore() {
            let mut vs = vs_arc.write()?; // Use write lock for zero_grad
            vs.zero_grad();
        }
        Ok(())
    }

    /// Returns all trainable parameters as ClioTensors.
    fn parameters(&self) -> Result<Vec<ClioTensor>> {
        match self.varstore() {
            Some(vs_arc) => {
                 let vs = vs_arc.read()?;
                 let params = vs.trainable_variables();
                 Ok(params.into_iter().map(ClioTensor::new).collect())
            },
            None => Ok(Vec::new()), // No parameters if no VarStore
        }
    }
}

/// Helper trait for PyO3 classes that wrap a Module.
/// Allows retrieving the underlying Module trait object.
/// This helps PySequential interact with layers in a type-safe Rust way.
pub trait PyModuleWrapper {
     // Provides dynamic access to the underlying Module trait implementation.
     // Use Box<dyn Module> for dynamic dispatch. Arc<dyn Module> might also work.
    fn get_module(&self) -> Box<dyn Module>;
     // Provides access back to the PyObject for Python-side operations if needed.
     // Requires the struct to hold its own Py<Self>.
    // fn py_object(&self) -> &PyObject;
}