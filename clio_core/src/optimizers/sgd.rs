use crate::error::{Result, ClioError};
use crate::tensor::{ClioTensor, py_to_clio_tensor};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (y_pred, y_true))]
pub fn mse_loss(y_pred: &Bound<'_, PyAny>, y_true: &Bound<'_, PyAny>) -> Result<ClioTensor> {
    let pred_tensor = py_to_clio_tensor(y_pred)?;
    let true_tensor = py_to_clio_tensor(y_true)?;

    let pred_inner = pred_tensor.lock()?;
    let true_inner = true_tensor.lock()?;

    if pred_inner.size() != true_inner.size() {
        return Err(ClioError::ShapeError { expected: true_inner.size(), got: pred_inner.size() });
    }
    if pred_inner.device() != true_inner.device() {
        return Err(ClioError::DeviceMismatch{ expected: true_inner.device(), got: pred_inner.device() });
    }

    let loss = pred_inner.mse_loss(&true_inner, tch::Reduction::Mean);
    Ok(ClioTensor::new(loss))
}

// TODO: Add cross_entropy_loss, etc.