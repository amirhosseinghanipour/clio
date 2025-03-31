use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError};
use pyo3::PyErr;
use thiserror::Error;
use tch::Device;

#[derive(Error, Debug)]
pub enum ClioError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] tch::TchError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Shape mismatch: Expected {expected:?}, got {got:?}")]
    ShapeError { expected: Vec<i64>, got: Vec<i64> },
    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Python interaction error: {0}")]
    PyConversionError(#[from] PyErr),
    #[error("Locking error: {0}")]
    LockError(String),
    #[error("Device mismatch: Expected {expected:?}, got {got:?}")]
    DeviceMismatch { expected: Device, got: Device },
    #[error("FFI error: {0}")]
    FfiError(String),
    #[error("Missing configuration: {0} must be set (e.g., via model.compile())")]
    MissingConfig(String),
    #[error("Operation not supported: {0}")]
    NotSupported(String),
     #[error("Type error: {0}")]
    TypeError(String),
}

// Conversion for lock poisoning errors
impl<T> From<std::sync::PoisonError<T>> for ClioError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        ClioError::LockError(err.to_string())
    }
}

// Conversion to PyErr for Python exceptions
impl From<ClioError> for PyErr {
    fn from(err: ClioError) -> PyErr {
        match err {
            ClioError::PyConversionError(py_err) => py_err,
            ClioError::ShapeError { .. }
            | ClioError::ConfigError(_)
            | ClioError::MissingConfig(_)
            | ClioError::DeviceMismatch { .. }
            | ClioError::ParameterNotFound(_)
            | ClioError::TypeError(_)
            | ClioError::NotSupported(_) => PyValueError::new_err(err.to_string()),
            ClioError::FfiError(_) => PyRuntimeError::new_err(format!("FFI Error: {}", err)),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, ClioError>;