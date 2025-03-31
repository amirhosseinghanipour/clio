use crate::error::{Result, ClioError};
use crate::tensor::{ClioTensor, py_to_clio_tensor};
use crate::device::{get_device, device_to_string};
use crate::modules::Module; // Import the Rust trait
use pyo3::prelude::*;
use tch::{nn, Device, Tensor};
use std::sync::{Arc, RwLock};

#[pyclass(name = "Linear")]
#[derive(Debug)]
pub struct PyLinear {
    vs: Arc<RwLock<nn::VarStore>>,
    linear: nn::Linear,
    in_dim: i64,
    out_dim: i64,
    // No need to store device separately, VarStore holds the authoritative device
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_dim, out_dim, use_bias=true, device=None))]
    fn new(in_dim: i64, out_dim: i64, use_bias: bool, device: Option<&str>) -> Result<Self> {
        let device = get_device(device)?;
        // Each layer starts with its own VS. It will be merged later by Sequential.
        let vs = Arc::new(RwLock::new(nn::VarStore::new(device)));
        let p = vs.read()?.root(); // Read lock is sufficient for path
        let config = nn::LinearConfig { bias: use_bias, ..Default::default() };
        let linear = nn::linear(&p / "linear", in_dim, out_dim, config);

        Ok(Self {
            vs,
            linear,
            in_dim,
            out_dim,
        })
    }

    /// Python-facing forward method.
    #[pyo3(signature = (xs))]
    fn forward(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        let input_tensor = py_to_clio_tensor(xs)?;
        // Call the internal Rust trait implementation
        <Self as Module>::forward(self, &input_tensor)
    }

    #[pyo3(signature = (xs))]
    fn __call__(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        self.forward(xs)
    }

    #[getter]
    fn weights(&self) -> Result<ClioTensor> {
        let vs = self.vs.read()?;
        match vs.get_var("linear.weight") {
            Some(t) => Ok(ClioTensor::new(t)),
            None => Err(ClioError::ParameterNotFound("linear.weight".into()))
        }
    }

    #[getter]
    fn bias(&self) -> Result<Option<ClioTensor>> {
        let vs = self.vs.read()?;
        match vs.get_var("linear.bias") {
            Some(t) => Ok(Some(ClioTensor::new(t))),
            None => Ok(None)
        }
    }

    /// Returns the VarStore (internal use, e.g., by Sequential).
    pub(crate) fn get_varstore_arc(&self) -> Arc<RwLock<nn::VarStore>> {
        self.vs.clone()
    }

    /// Python-facing device move.
    #[pyo3(signature = (device_str))]
    fn to(&self, device_str: &str) -> Result<()> {
        let new_device = get_device(Some(device_str))?;
         <Self as Module>::to(self, new_device)
    }

     #[getter]
     fn parameters(&self) -> Result<Vec<ClioTensor>> {
         <Self as Module>::parameters(self)
     }


    fn __str__(&self) -> String {
         let device = self.vs.read().map(|vs| vs.device()).unwrap_or(Device::Cpu); // Best effort read
        format!("Linear(in_dim={}, out_dim={}, bias={}, device='{}')",
                self.in_dim, self.out_dim, self.bias().map(|b| b.is_some()).unwrap_or(false), device_to_string(device))
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

// Implement the internal Rust Module trait
impl Module for PyLinear {
    fn forward(&self, xs: &ClioTensor) -> Result<ClioTensor> {
        let xs_inner = xs.lock()?;
        let vs_device = self.vs.read()?.device(); // Read lock sufficient
        if xs_inner.device() != vs_device {
            return Err(ClioError::DeviceMismatch{ expected: vs_device, got: xs_inner.device() });
        }
         // Basic shape check
         let expected_last_dim = self.in_dim;
         let actual_shape = xs_inner.size();
         if actual_shape.is_empty() || *actual_shape.last().unwrap() != expected_last_dim {
              return Err(ClioError::ShapeError { expected: vec![-1, expected_last_dim], got: actual_shape });
         }

        // Use tch's ModuleT::forward
        let output = self.linear.forward(&*xs_inner);
        Ok(ClioTensor::new(output))
    }

    fn varstore(&self) -> Option<Arc<RwLock<nn::VarStore>>> {
        Some(self.vs.clone())
    }

    fn to(&self, device: Device) -> Result<()> {
        let current_device = self.vs.read()?.device();
         if device != current_device {
             // Use write lock to modify the VarStore's device
             self.vs.write()?.set_device(device);
              println!("Linear layer moved to {:?}", device);
         }
        Ok(())
    }
    // zero_grad and parameters provided by default Module impl
}