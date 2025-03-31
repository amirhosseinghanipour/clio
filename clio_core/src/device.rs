use crate::error::{Result, ClioError};
use tch::Device;

pub fn get_device(device_str: Option<&str>) -> Result<Device> {
    match device_str {
        Some("cuda") => {
            if tch::utils::has_cuda() { Ok(Device::Cuda(0)) }
            else { Err(ClioError::ConfigError("CUDA selected, but not available.".to_string())) }
        },
        Some("mps") => {
             if tch::utils::has_mps() { Ok(Device::Mps) }
             else { Err(ClioError::ConfigError("MPS selected, but not available.".to_string())) }
        },
        Some("cpu") => Ok(Device::Cpu),
        Some(other) => Err(ClioError::ConfigError(format!("Unsupported device string: '{}'", other))),
        None => {
            if tch::utils::has_cuda() { Ok(Device::Cuda(0)) }
            else if tch::utils::has_mps() { Ok(Device::Mps) }
            else { Ok(Device::Cpu) }
        }
    }
}

pub fn device_to_string(device: Device) -> String {
     match device {
         Device::Cpu => "cpu".to_string(),
         Device::Cuda(_) => "cuda".to_string(),
         Device::Mps => "mps".to_string(),
     }
}