use crate::error::{Result, ClioError};
use crate::tensor::{ClioTensor, py_to_clio_tensor};
use crate::device::{get_device, device_to_string};
use crate::modules::Module;
use pyo3::prelude::*;
use tch::{nn, Device};
use std::sync::{Arc, RwLock}; // Keep imports for consistency

#[pyclass(name = "ReLU")]
#[derive(Debug, Clone)] // Clone might be useful
pub struct PyReLU {
     // Store intended device, although operation is device-agnostic once input is correct
     device: Device,
}

#[pymethods]
impl PyReLU {
    #[new]
    #[pyo3(signature = (device = None))]
    fn new(device: Option<&str>) -> Result<Self> {
        Ok(Self { device: get_device(device)? })
    }

    #[pyo3(signature = (xs))]
    fn forward(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        let input_tensor = py_to_clio_tensor(xs)?;
        <Self as Module>::forward(self, &input_tensor)
    }

    #[pyo3(signature = (xs))]
    fn __call__(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
         self.forward(xs)
    }

     /// Python-facing device move (updates internal tracker).
    #[pyo3(signature = (device_str))]
    fn to(&mut self, device_str: &str) -> Result<()> {
         let new_device = get_device(Some(device_str))?;
         <Self as Module>::to(self, new_device)?; // Call Rust trait version
         self.device = new_device; // Update internal state
         Ok(())
    }

     #[getter]
     fn parameters(&self) -> Result<Vec<ClioTensor>> {
         <Self as Module>::parameters(self)
     }


    fn __str__(&self) -> String {
        format!("ReLU(device='{}')", device_to_string(self.device))
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl Module for PyReLU {
    fn forward(&self, xs: &ClioTensor) -> Result<ClioTensor> {
        let xs_inner = xs.lock()?;
        // Check device consistency as good practice
         if xs_inner.device() != self.device {
             return Err(ClioError::DeviceMismatch{ expected: self.device, got: xs_inner.device() });
         }
        let output = xs_inner.relu();
        Ok(ClioTensor::new(output))
    }

    fn varstore(&self) -> Option<Arc<RwLock<nn::VarStore>>> {
        None // ReLU has no parameters/VarStore
    }

    fn to(&self, device: Device) -> Result<()> {
         // This conceptually needs mutable access to self.device,
         // but Module trait expects &self. The Python `to` method handles mutation.
         // This Rust trait method only needs to confirm device compatibility or no-op.
         println!("ReLU layer 'to' called for device {:?} (internal device tracker updated via Python method)", device);
        Ok(())
    }
    // zero_grad and parameters provided by default Module impl (will be empty)
}