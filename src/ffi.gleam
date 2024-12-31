// Binding to the native functions in the rust ortex library via erlang.
// As the result type for Nif functions is not compatible with the
// gleam result type, it has to be converted.
//
// Most functions have a nif ffi function (nif_...) and a public function
// that converts the reuslt type and potentially also paramters.
import gleam/dynamic
import gleam/erlang/atom.{type Atom}
import gleam/option.{type Option}
import gluple/reflect.{list_to_tuple}

// opaque types, that are pure ort types
pub type OrtModel

pub type OrtTensor

pub type NifResult(val, err)

@external(erlang, "Elixir.Ortex.Native", "nif_result_to_result")
fn nif_result_to_result(nif: NifResult(val, err)) -> Result(val, err)

@external(erlang, "Elixir.Ortex.Native", "init")
fn init_nif(
  path: String,
  eps: List(Atom),
  opt: Int,
) -> NifResult(OrtModel, String)

pub fn init(path: String, eps: List(Atom), opt: Int) -> Result(OrtModel, String) {
  init_nif(path, eps, opt) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "run")
fn run_nif(
  model: OrtModel,
  inputs: List(OrtTensor),
) -> NifResult(List(#(OrtTensor, List(Int), atom.Atom, Int)), String)

pub fn run(
  model: OrtModel,
  inputs: List(OrtTensor),
) -> Result(List(#(OrtTensor, List(Int), atom.Atom, Int)), String) {
  run_nif(model, inputs) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "show_ession")
fn show_session_nif(
  model: OrtModel,
) -> NifResult(
  #(
    List(#(String, String, Option(List(Int)))),
    List(#(String, String, Option(List(Int)))),
  ),
  String,
)

pub fn show_session(
  model: OrtModel,
) -> Result(
  #(
    List(#(String, String, Option(List(Int)))),
    List(#(String, String, Option(List(Int)))),
  ),
  String,
) {
  show_session_nif(model) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "from_binary")
fn from_binary_nif(
  bin: BitArray,
  shape: dynamic.Dynamic,
  dtype: #(atom.Atom, Int),
) -> NifResult(OrtTensor, String)

pub fn from_binary(bin: BitArray, shape: List(Int), dtype: #(atom.Atom, Int)) {
  let nif_result = from_binary_nif(bin, list_to_tuple(shape), dtype)
  let result = nif_result_to_result(nif_result)
  result
}

@external(erlang, "Elixir.Ortex.Native", "to_binary")
fn to_binary_nif(
  tensor: OrtTensor,
  bits: Int,
  limit: Int,
) -> NifResult(BitArray, String)

pub fn to_binary(
  tensor: OrtTensor,
  bits: Int,
  limit: Int,
) -> Result(BitArray, String) {
  to_binary_nif(tensor, bits, limit) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "slice")
fn slice_nif(
  tensor: OrtTensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> NifResult(OrtTensor, String)

pub fn slice(
  tensor: OrtTensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> Result(OrtTensor, String) {
  slice_nif(tensor, start_indicies, lengths, strides) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "reshape")
fn reshape_nif(
  tensor: OrtTensor,
  shape: dynamic.Dynamic,
) -> NifResult(OrtTensor, String)

pub fn reshape(tensor: OrtTensor, shape: List(Int)) -> Result(OrtTensor, String) {
  reshape_nif(tensor, list_to_tuple(shape)) |> nif_result_to_result
}

@external(erlang, "Elixir.Ortex.Native", "concatenate")
fn concatenate_nif(
  tensors: List(OrtTensor),
  dtype: atom.Atom,
  axis: Int,
) -> NifResult(OrtTensor, String)

pub fn concatenate(tensors: List(OrtTensor), dtype: atom.Atom, axis: Int) {
  concatenate_nif(tensors, dtype, axis) |> nif_result_to_result
}
