use crate::error::{Result, ClioError};
use crate::tensor::{ClioTensor, py_to_clio_tensor};
use crate::device::{get_device, device_to_string};
use crate::modules::Module;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};
use tch::{nn, Device, Tensor};
use std::sync::{Arc, RwLock};
use std::path::Path;

#[pyclass(name = "Sequential")]
#[derive(Debug)]
pub struct PySequential {
    // Store Python layer objects for easy Python interaction & method calls
    layers: Vec<PyObject>,
    // A single, unified VarStore for the entire model.
    vs: Arc<RwLock<nn::VarStore>>,
    // Track the device the *model* (and its VarStore) lives on
    device: Device,

    // --- Training Loop State ---
    optimizer: Option<PyObject>, // Stores the compiled Python optimizer object
    loss: Option<PyObject>,      // Stores the compiled Python loss function object
    is_compiled: bool,
}

#[pymethods]
impl PySequential {
    #[new]
    #[pyo3(signature = (layers=None, device=None))]
    fn new(layers: Option<&Bound<'_, PyList>>, device: Option<&str>) -> Result<Self> {
        let device = get_device(device)?;
        let model_vs = Arc::new(RwLock::new(nn::VarStore::new(device)));

        let mut slf = Self {
            layers: Vec::new(), // Initialize empty, add layers below
            vs: model_vs,
            device,
            optimizer: None,
            loss: None,
            is_compiled: false,
        };

        if let Some(layers_list) = layers {
            for layer_obj in layers_list.iter() {
                // Borrow layer_obj to pass PyObject
                slf.add(layer_obj.to_object(layers_list.py()))?;
            }
        }

        Ok(slf)
    }

    /// Adds a layer to the model, merging its parameters into the model's VarStore.
    fn add(&mut self, layer: PyObject) -> Result<()> {
        let py = layer.py();
        let layer_ref = layer.bind(py); // Bound reference for method calls

        // 1. Determine the path prefix for this layer
        let layer_prefix = format!("layer_{}", self.layers.len());

        // 2. Attempt to extract the layer's original VarStore (if it's a known Clio type)
        //    We need a way to identify Clio layers and access their Rust internals.
        //    Using hasattr for `get_varstore_arc` is a convention-based approach.
        let mut maybe_layer_vs_arc: Option<Arc<RwLock<nn::VarStore>>> = None;
        if let Ok(method) = layer_ref.getattr("get_varstore_arc") {
             if let Ok(arc_method) = method.downcast::<pyo3::types::PyCFunction>() {
                // Directly call the Rust function via PyO3 internals is tricky/unstable.
                // Alternative: Try extracting the PyLinear/PyReLU directly.
                 if let Ok(py_linear) = layer_ref.extract::<PyRefMut<super::linear::PyLinear>>() {
                    maybe_layer_vs_arc = Some(py_linear.get_varstore_arc());
                 } else if let Ok(_py_relu) = layer_ref.extract::<PyRefMut<super::activations::PyReLU>>() {
                    // ReLU has no VS, so `None` is expected/fine.
                    maybe_layer_vs_arc = None;
                 }
                 // Add checks for other Clio layer types here...
                 else {
                    println!("Warning: Layer type {:?} not recognized for VarStore merging.", layer_ref.get_type().qualname()?);
                }
             }
        }

        // 3. Merge Parameters (if a VarStore was found)
        if let Some(layer_vs_arc) = maybe_layer_vs_arc {
             let layer_vs = layer_vs_arc.read()?; // Read lock source VS
             let mut model_vs = self.vs.write()?; // Write lock target VS

             // Move trainable variables
             for (name, var) in layer_vs.trainable_variables_with_names() {
                let new_name = format!("{}.{}", layer_prefix, name);
                // Check if name already exists (shouldn't happen with unique prefix)
                if model_vs.get_var(&new_name).is_some() {
                    return Err(ClioError::ConfigError(format!("Duplicate parameter name during merge: {}", new_name)));
                }
                 // Use `add_var` or similar mechanism if VarStore has one,
                 // or rely on `move_var` (which might require creating it first if target doesn't exist)
                 // `move_var` seems appropriate here. Ensure correct path handling.
                 // `tch` VarStore doesn't have a direct `move_var` between stores.
                 // We need to iterate, get tensors, and add them to the new VS.
                 // This is complex if we want zero-copy.
                 // Simpler (but involves copy/recreation): Add variables to the model VS.
                 // We need access to the original tensor creation logic for this.

                 // WORKAROUND: Let's assume for now that layers *will be constructed*
                 // with the model's VarStore and the correct path prefix.
                 // This shifts responsibility to the user or a future factory pattern.
                 // The `.add` method will primarily ensure device consistency.
                 println!("Warning: VarStore merging in Sequential.add is NOT yet implemented by moving variables.");
                 println!("         Layers should ideally be constructed using the model's VarStore path.");

                 // Ensure layer is on the correct device *now*
                 if var.device() != self.device {
                     // Attempt to move the layer via its 'to' method
                     if let Ok(to_method) = layer_ref.getattr("to") {
                          let device_str = device_to_string(self.device);
                          to_method.call1((device_str,))?;
                     } else {
                          return Err(ClioError::ConfigError(format!(
                             "Layer {} parameter '{}' is on {:?} but model is on {:?}, and layer has no 'to' method.",
                             layer_prefix, name, var.device(), self.device
                          )));
                     }
                 }
             }
        } else {
             // If no VarStore found (e.g., ReLU or external Python layer),
             // just ensure device consistency if possible.
             if let Ok(to_method) = layer_ref.getattr("to") {
                 let device_str = device_to_string(self.device);
                 to_method.call1((device_str,))?;
             }
        }


        // 4. Add the Python object to the list
        self.layers.push(layer);
        Ok(())
    }


    /// Python-facing forward method.
    #[pyo3(signature = (xs))]
    fn forward(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        let mut current_tensor = py_to_clio_tensor(xs)?; // Start with ClioTensor

        for layer_obj in &self.layers {
            let layer = layer_obj.bind(xs.py()); // Get Bound reference
            // Call the layer's forward method
            let output_any = layer.call_method1("forward", (current_tensor,))?; // Pass ClioTensor
            current_tensor = py_to_clio_tensor(&output_any)?; // Convert result back
        }
        Ok(current_tensor)
    }

    #[pyo3(signature = (xs))]
    fn __call__(&self, xs: &Bound<'_, PyAny>) -> Result<ClioTensor> {
        self.forward(xs)
    }

    // --- Training API Methods ---

    #[pyo3(signature = (optimizer, loss, metrics=None))]
    fn compile(&mut self, optimizer: PyObject, loss: PyObject, _metrics: Option<Vec<PyObject>>) -> Result<()> {
        // TODO: Validate optimizer and loss types?
        self.optimizer = Some(optimizer);
        self.loss = Some(loss);
        // TODO: Handle metrics
        self.is_compiled = true;
         println!("Model compiled.");
        Ok(())
    }

    #[pyo3(signature = (x, y, batch_size=32, epochs=1, verbose=1, validation_data=None))]
    fn fit(&self, x: PyObject, y: PyObject, batch_size: usize, epochs: usize, verbose: usize, _validation_data: Option<&Bound<'_, PyAny>>) -> Result<PyObject> { // TODO: Use validation_data
        let py = x.py(); // Get Python instance from one of the inputs

        // --- Pre-checks ---
        if !self.is_compiled {
            return Err(ClioError::MissingConfig("optimizer and loss (call model.compile first)".into()));
        }
        let optimizer = self.optimizer.as_ref().ok_or_else(|| ClioError::MissingConfig("optimizer".into()))?.bind(py);
        let loss_fn = self.loss.as_ref().ok_or_else(|| ClioError::MissingConfig("loss".into()))?.bind(py);

        // --- Data Handling (Basic - Assumes PyTorch tensors for now) ---
        // Need a Clio DataLoader or similar for robust batching & parallelism
        let x_bound = x.bind(py);
        let y_bound = y.bind(py);
        let num_samples = x_bound.len()?;
        if num_samples == 0 { return Err(ClioError::ConfigError("Input data x cannot be empty.".into()));}
        if num_samples != y_bound.len()? { return Err(ClioError::ShapeError{ expected: vec![num_samples as i64], got: vec![y_bound.len()? as i64]});}

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        println!("Starting training on {} samples, batch size {}, {} epochs...", num_samples, batch_size, epochs);

        // --- Training Loop ---
         let mut history = Vec::<(usize, f64)>::new(); // Store (epoch, avg_loss)

        for epoch in 0..epochs {
            if verbose > 0 { println!("Epoch {}/{}", epoch + 1, epochs); }
            let mut epoch_loss_sum = 0.0;
            let mut batch_count = 0;

            // Basic batch slicing (replace with DataLoader)
            for i in 0..num_batches {
                let start = i * batch_size;
                let end = (start + batch_size).min(num_samples);
                if start >= end { continue; } // Should not happen with correct logic

                // Get batch data (as PyTorch tensors via slicing)
                // Slicing Python objects requires care
                let slice = PySlice::new_bound(py, start as isize, end as isize, 1);
                let batch_x_obj = x_bound.get_item(&slice)?;
                let batch_y_obj = y_bound.get_item(&slice)?;

                // Convert batch to ClioTensors using FFI bridge
                let batch_x = py_to_clio_tensor(&batch_x_obj)?;
                let batch_y = py_to_clio_tensor(&batch_y_obj)?;

                // --- Autograd Step ---
                // 1. Zero gradients (call optimizer's zero_grad)
                optimizer.call_method0("zero_grad")?;

                // 2. Forward pass (use model's forward)
                let predictions = self.forward(&batch_x_obj)?; // Pass PyAny/PyObject is okay here

                // 3. Calculate loss (call Python loss function)
                let loss_clio = loss_fn.call1((&predictions, &batch_y))?; // Pass ClioTensor args
                let loss_clio = py_to_clio_tensor(&loss_clio)?; // Ensure result is ClioTensor

                // 4. Backward pass (on the ClioTensor loss)
                loss_clio.backward()?;

                // 5. Optimizer step (call optimizer's step)
                optimizer.call_method0("step")?;
                // --- End Autograd Step ---

                // --- Logging ---
                 let loss_val = match loss_clio.item() {
                    Ok(item_obj) => item_obj.extract::<f64>(py).unwrap_or(f64::NAN),
                    Err(_) => f64::NAN, // Handle error case if item() fails
                 };

                 if !loss_val.is_nan() {
                    epoch_loss_sum += loss_val;
                    batch_count += 1;
                 }

                 if verbose > 0 {
                     let avg_loss_so_far = if batch_count > 0 { epoch_loss_sum / batch_count as f64 } else { 0.0 };
                     // Use simple print, avoid \r for clarity in logs
                     if (i + 1) % ((num_batches / 10).max(1)) == 0 || (i+1) == num_batches { // Print ~10 times per epoch
                       println!("  Batch {}/{} - loss: {:.4}", i + 1, num_batches, avg_loss_so_far);
                     }
                 }
            } // End batch loop

            let avg_epoch_loss = if batch_count > 0 { epoch_loss_sum / batch_count as f64 } else { 0.0 };
             if verbose > 0 {
                 println!("Epoch {} Average Loss: {:.4}", epoch + 1, avg_epoch_loss);
             }
             history.push((epoch + 1, avg_epoch_loss));

            // TODO: Handle validation_data

        } // End epoch loop

        // Return history object (simple dict for now)
        let history_dict = pyo3::types::PyDict::new_bound(py);
        history_dict.set_item("epoch", history.iter().map(|(e, _)| *e).collect::<Vec<_>>())?;
        history_dict.set_item("loss", history.iter().map(|(_, l)| *l).collect::<Vec<_>>())?;
        Ok(history_dict.into())
    }


    #[pyo3(signature = (x, y, batch_size=32, verbose=1))]
    fn evaluate(&self, x: PyObject, y: PyObject, batch_size: usize, verbose: usize) -> Result<PyObject> {
        let py = x.py();
        // --- Pre-checks ---
        if !self.is_compiled {
             return Err(ClioError::MissingConfig("loss function (call model.compile first)".into()));
        }
         let loss_fn = self.loss.as_ref().ok_or_else(|| ClioError::MissingConfig("loss".into()))?.bind(py);

        // --- Data Handling ---
         let x_bound = x.bind(py);
         let y_bound = y.bind(py);
         let num_samples = x_bound.len()?;
         if num_samples == 0 { return Err(ClioError::ConfigError("Input data x cannot be empty.".into()));}
         if num_samples != y_bound.len()? { return Err(ClioError::ShapeError{ expected: vec![num_samples as i64], got: vec![y_bound.len()? as i64]});}
         let num_batches = (num_samples + batch_size - 1) / batch_size;

         if verbose > 0 { println!("Evaluating on {} samples...", num_samples); }

        let mut total_loss_sum = 0.0;
        let mut total_batches_evaluated = 0;

        // --- Evaluation Loop (No Gradients) ---
        let _guard = tch::no_grad_guard(); // <<< Disable gradients

        for i in 0..num_batches {
             let start = i * batch_size;
             let end = (start + batch_size).min(num_samples);
             if start >= end { continue; }

             let slice = PySlice::new_bound(py, start as isize, end as isize, 1);
             let batch_x_obj = x_bound.get_item(&slice)?;
             let batch_y_obj = y_bound.get_item(&slice)?;

             let batch_x = py_to_clio_tensor(&batch_x_obj)?;
             let batch_y = py_to_clio_tensor(&batch_y_obj)?;

             let predictions = self.forward(&batch_x_obj)?;
             let loss_clio = loss_fn.call1((&predictions, &batch_y))?;
             let loss_clio = py_to_clio_tensor(&loss_clio)?;

             if let Ok(item_obj) = loss_clio.item() {
                if let Ok(loss_val) = item_obj.extract::<f64>(py) {
                   total_loss_sum += loss_val;
                   total_batches_evaluated += 1;
                }
             }
             if verbose > 0 && (i + 1) % ((num_batches / 10).max(1)) == 0 || (i+1) == num_batches {
                  println!("  Evaluated Batch {}/{}", i + 1, num_batches);
             }
        }

        let avg_loss = if total_batches_evaluated > 0 { total_loss_sum / total_batches_evaluated as f64 } else { f64::NAN };
         if verbose > 0 {
             println!("Evaluation Complete. Average Loss: {:.4}", avg_loss);
         }

         Ok(avg_loss.to_object(py)) // Return avg loss as float
    }


    #[pyo3(signature = (x, batch_size=32, verbose=0))]
    fn predict(&self, x: PyObject, batch_size: usize, verbose: usize) -> Result<ClioTensor> {
        let py = x.py();
         // --- Data Handling ---
         let x_bound = x.bind(py);
         let num_samples = x_bound.len()?;
         if num_samples == 0 { return Err(ClioError::ConfigError("Input data x cannot be empty.".into()));}
         let num_batches = (num_samples + batch_size - 1) / batch_size;

         if verbose > 0 { println!("Predicting on {} samples...", num_samples); }

         let mut all_predictions_list: Vec<Tensor> = Vec::with_capacity(num_batches);

         // --- Prediction Loop (No Gradients) ---
         let _guard = tch::no_grad_guard();

        for i in 0..num_batches {
             let start = i * batch_size;
             let end = (start + batch_size).min(num_samples);
             if start >= end { continue; }

             let slice = PySlice::new_bound(py, start as isize, end as isize, 1);
             let batch_x_obj = x_bound.get_item(&slice)?;
             // let batch_x = py_to_clio_tensor(&batch_x_obj)?; // Not needed if forward takes PyAny

             let predictions_clio = self.forward(&batch_x_obj)?;

             // Collect underlying tch::Tensors for concatenation
             all_predictions_list.push(predictions_clio.lock()?.shallow_clone()); // Clone tensor inside mutex guard

             if verbose > 0 && (i + 1) % ((num_batches / 10).max(1)) == 0 || (i+1) == num_batches {
                  println!("  Predicted Batch {}/{}", i + 1, num_batches);
             }
        }

        // Concatenate results using tch::Tensor::cat
         if all_predictions_list.is_empty() {
             Err(ClioError::ConfigError("Prediction yielded no results.".into()))
         } else {
             let tensors_to_cat: Vec<_> = all_predictions_list.iter().collect(); // Collect refs
             let concatenated = Tensor::cat(&tensors_to_cat, 0)?;
             Ok(ClioTensor::new(concatenated))
         }
    }

    // --- Parameter Access & Management ---
    #[getter]
    fn parameters(&self) -> Result<Vec<ClioTensor>> {
         <Self as Module>::parameters(self)
    }

    /// Provides the VarStore Arc address (for internal use by Optimizers initially).
    /// Marked pub(crate) as it's an internal detail.
    pub(crate) fn get_varstore_arc_address(&self) -> usize {
        Arc::as_ptr(&self.vs) as usize
    }

     /// Python-facing device move.
    #[pyo3(signature = (device_str))]
    fn to(&mut self, device_str: &str) -> Result<()> {
        let new_device = get_device(Some(device_str))?;
         if new_device != self.device {
            println!("Moving Sequential model to {:?}", new_device);
             // Move own VarStore first
             self.vs.write()?.set_device(new_device);
             self.device = new_device;

             // Move individual layers
             let py = Python::acquire_gil().python();
             for layer_obj in &self.layers {
                 let layer_bound = layer_obj.bind(py);
                 if let Ok(to_method) = layer_bound.getattr("to") {
                      // Pass &mut self if layer's `to` takes it
                     if let Ok(slf) = layer_bound.extract::<PyRefMut<PyAny>>() {
                        // Need a way to call `to` on PyRefMut if it's defined that way
                         to_method.call1((device_str,))?; // Assume it takes &self or &mut self appropriately
                     } else {
                          // Fallback if extract fails
                         to_method.call1((device_str,))?;
                     }

                 } else {
                     println!("Warning: Layer {:?} does not have 'to' method for device transfer.", layer_bound.get_type().qualname()?);
                 }
             }
         }
        Ok(())
    }

    // --- Serialization ---
    #[pyo3(signature = (path))]
    fn save_weights(&self, path: &str) -> Result<()> {
        self.vs.read()?.save(Path::new(path))?;
         println!("Model weights saved to: {}", path);
        Ok(())
    }

    #[pyo3(signature = (path))]
    fn load_weights(&self, path: &str) -> Result<()> {
         // Check device consistency? VarStore::load might handle this.
        self.vs.write()?.load(Path::new(path))?;
         println!("Model weights loaded from: {}", path);
        Ok(())
    }

    fn __str__(&self) -> String {
         let py = Python::acquire_gil().python();
         let layer_strs: Vec<String> = self.layers.iter().map(|layer_obj| {
             layer_obj.bind(py).str().map_or_else(|_| "<Layer>".to_string(), |s| s.to_string())
         }).collect();
         format!("Sequential(device='{}', compiled={}, layers=[\n  {}\n])",
                 device_to_string(self.device), self.is_compiled, layer_strs.join(",\n  "))
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
}


 // Implement the internal Rust Module trait
impl Module for PySequential {
    fn forward(&self, xs: &ClioTensor) -> Result<ClioTensor> {
         Err(ClioError::NotSupported("Internal Rust forward pass not supported for PySequential containing Python objects. Use Python API.".into()))
    }

    fn varstore(&self) -> Option<Arc<RwLock<nn::VarStore>>> {
        Some(self.vs.clone())
    }

     fn to(&self, device: Device) -> Result<()> {
         // This requires mutable access to self.device and layers.
         // Rely on the Python `to` method which takes `&mut self`.
         println!("PySequential::Module::to called for {:?} (use Python `model.to()` for actual move)", device);
         Ok(())
     }
    // zero_grad and parameters provided by default Module impl using the model's vs
}