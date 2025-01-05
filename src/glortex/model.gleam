import ffi
import gleam/erlang/atom
import gleam/list
import gleam/result
import glortex/dtype
import glortex/tensor.{type Tensor}

pub type Model {
  Model(ffi.OrtModel)
}

//Load an `Ortex.Model` from disk. Pass the execution providers as a list,
// of descending priority (pass empty list for cpu) and graph optimization level 1-3.
// Any graph optimization level beyond the range of 1-3 will disable graph optimization.
// 
// For details also see the documentation in ortex here:
//   https://github.com/elixir-nx/ortex/blob/main/lib/ortex.ex
pub fn load(
  path: String,
  eps: List(atom.Atom),
  opt: Int,
) -> Result(Model, String) {
  let eps = case eps {
    [] -> [atom.create_from_string("cpu")]
    _ -> eps
  }
  use ort_model <- result.try(ffi.init(path, eps, opt))

  Ok(Model(ort_model))
}

pub fn run(model: Model, inputs: List(Tensor)) -> Result(List(Tensor), String) {
  let Model(ort_model) = model

  let inputs = list.map(inputs, fn(input) { input.data })

  use result <- result.try(ffi.run(ort_model, inputs))

  Ok(
    list.map(result, fn(data) {
      let #(ort_tensor, dims, t, prec) = data
      let assert Ok(t) = dtype.from_type_and_prec(t, prec)
      tensor.Tensor(ort_tensor, dims, t)
    }),
  )
}

pub fn show_session(model: Model) {
  let Model(ort_model) = model
  ffi.show_session(ort_model)
}
