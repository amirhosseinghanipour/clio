use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod error;
mod device;
mod tensor;
mod modules;
mod optimizers;

use tensor::ClioTensor;
use modules::{PyLinear, PyReLU, PySequential};
use optimizers::PySGD;
use losses::mse_loss;


#[pymodule]
#[pyo3(name = "_core")]
fn clio_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ClioTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PySGD>()?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;

    // Add helpers or constants
    m.add("has_cuda", tch::utils::has_cuda())?;
    m.add("has_mps", tch::utils::has_mps())?;

    Ok(())
}