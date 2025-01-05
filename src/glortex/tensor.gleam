import ffi.{type OrtTensor}
import gleam/bool
import gleam/erlang/atom
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import glortex/dtype.{type Dtype}

@external(erlang, "binary", "copy")
fn copy(data: BitArray, n: Int) -> BitArray

pub type Tensor {
  Tensor(data: OrtTensor, shape: List(Int), dtype: Dtype)
}

// Create a tensor of floats with given precision and size with the given value
// repeated.
pub fn broadcast_float(
  value: Float,
  precision: Int,
  shape: List(Int),
) -> Result(Tensor, String) {
  use t <- result.try(dtype.from_type_and_prec(
    atom.create_from_string("f"),
    precision,
  ))

  let data =
    copy(
      <<value:float-size(precision)-native>>,
      list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
    )

  use ort_tensor <- result.try(
    ffi.from_binary(data, shape, #(atom.create_from_string("f"), precision)),
  )

  Ok(Tensor(ort_tensor, shape, t))
}

// create a tensor from binary
pub fn from_binary(
  data: BitArray,
  shape: List(Int),
  t: Dtype,
) -> Result(Tensor, String) {
  use ort_tensor <- result.try(ffi.from_binary(data, shape, dtype.type_tuple(t)))

  Ok(Tensor(ort_tensor, shape, t))
}

// get the binary data from a tensor
pub fn to_binary(tensor: Tensor, limit: Int) -> Result(BitArray, String) {
  let Tensor(ort_tensor, _shape, t) = tensor
  use res <- result.try(ffi.to_binary(ort_tensor, dtype.precision(t), limit))

  Ok(res)
}

// Create an list of floats from binary data with the given precision
fn binary_to_floats(
  binary: BitArray,
  precision: Int,
) -> Result(List(Float), String) {
  case binary {
    <<>> -> Ok([])
    <<a:float-32-native, rest:bits>> if precision == 32 -> {
      use tail <- result.try(binary_to_floats(rest, 32))
      Ok([a, ..tail])
    }

    <<a:float-64-native, rest:bits>> if precision == 64 -> {
      use tail <- result.try(binary_to_floats(rest, 64))
      Ok([a, ..tail])
    }

    _ if precision != 32 && precision != 64 -> Error("impossible precision")
    _ ->
      Error(
        "BitString connaot be converted to float list with precision "
        |> string.append(int.to_string(precision)),
      )
  }
}

// Reshape a tensor
pub fn reshape(tensor: Tensor, shape: List(Int)) -> Result(Tensor, String) {
  let Tensor(ort_tensor, _old_shape, dtype) = tensor

  use reshaped_ort_tensor <- result.try(ffi.reshape(ort_tensor, shape))

  Ok(Tensor(reshaped_ort_tensor, shape, dtype))
}

// Flatten a tensor, creating a tensor of same data size with only one dimension
pub fn flatten(tensor: Tensor) -> Result(Tensor, String) {
  let Tensor(_ort_tensor, shape, _dtype) = tensor

  case list.length(shape) {
    1 -> {
      Ok(tensor)
    }
    _ ->
      reshape(tensor, [
        list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
      ])
  }
}

fn all_equal(vs: List(Int)) -> Bool {
  case vs {
    [v0, v1, ..vs] ->
      case v0 == v1 {
        True -> all_equal([v1, ..vs])
        False -> False
      }
    _ -> True
  }
}

// Concatenate 2 shapes
fn concatenate_two_shapes(
  shape1: List(Int),
  shape2: List(Int),
  axis: Int,
) -> Result(List(Int), String) {
  case shape1, shape2 {
    [h1, ..shape1], [h2, ..shape2] -> {
      use rest_conc <- result.try(concatenate_two_shapes(
        shape1,
        shape2,
        axis - 1,
      ))
      case axis {
        0 -> Ok([h1 + h2, ..rest_conc])
        _ if h1 == h2 -> Ok([h1, ..rest_conc])
        _ -> Error("shapes can not be concated along the given axis")
      }
    }
    [], [] -> Ok([])
    _, _ -> Error("shapes can not be concated along the given axis")
  }
}

// Helper function, for computing the shape of a tensor
// that would result in concatenating tensors of given `shapes`
// along a given dimension (`axis`).
fn concatenate_shapes(
  shapes: List(List(Int)),
  axis: Int,
) -> Result(List(Int), String) {
  case shapes {
    [s1, s2, ..shapes] ->
      case concatenate_two_shapes(s1, s2, axis) {
        Ok(res) -> concatenate_shapes([res, ..shapes], axis)
        e -> e
      }
    [s1] -> Ok(s1)
    [] -> Error("cannot concatenate empty list of shapes")
  }
}

// Concatenate a list of `tensors` along a given dimension (`axis`).
pub fn concatenate(tensors: List(Tensor), axis: Int) -> Result(Tensor, String) {
  case tensors {
    [] -> Error("cannot concatenate empty list if tensors")
    [Tensor(_tensor0, _shape0, dtype0), ..] -> {
      let ort_tensors = list.map(tensors, fn(t) { t.data })
      let shapes = list.map(tensors, fn(t) { t.shape })
      let dtypes = list.map(tensors, fn(t) { t.dtype })

      use <- bool.guard(
        list.any(dtypes, fn(d) { d != dtype0 }),
        Error("Input dtypes must match"),
      )

      use shape <- result.try(concatenate_shapes(shapes, axis))
      use tensor <- result.try(ffi.concatenate(
        ort_tensors,
        dtype.type_atom(dtype0),
        axis,
      ))

      Ok(Tensor(tensor, shape, dtype0))
    }
  }
}

// Convert `tensor` of floats into a list of floats.
// `limit` is the maximum number of bytes in the data,
// which is needed by ortex.
pub fn to_float_list(tensor: Tensor, limit: Int) -> Result(List(Float), String) {
  use Tensor(flattened_ort_tensor, _, t) <- result.try(flatten(tensor))

  let prec = dtype.precision(t)

  use bin <- result.try(ffi.to_binary(flattened_ort_tensor, prec, limit))

  binary_to_floats(bin, prec)
}
