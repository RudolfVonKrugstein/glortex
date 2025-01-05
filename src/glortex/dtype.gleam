import gleam/erlang/atom
import gleam/int
import gleam/string

pub type Dtype {
  SignedInt8
  SignedInt16
  SignedInt32
  SignedInt64
  UnsignedInt8
  UnsignedInt16
  UnsignedInt32
  UnsignedInt64
  Float16
  Float32
  Float64
}

pub fn type_atom(dtype: Dtype) {
  case dtype {
    SignedInt8 | SignedInt16 | SignedInt32 | SignedInt64 ->
      atom.create_from_string("s")
    UnsignedInt8 | UnsignedInt16 | UnsignedInt32 | UnsignedInt64 ->
      atom.create_from_string("u")
    Float16 | Float32 | Float64 -> atom.create_from_string("f")
  }
}

pub fn precision(dtype: Dtype) {
  case dtype {
    SignedInt8 | UnsignedInt8 -> 8
    SignedInt16 | UnsignedInt16 | Float16 -> 16
    SignedInt32 | UnsignedInt32 | Float32 -> 32
    SignedInt64 | UnsignedInt64 | Float64 -> 64
  }
}

pub fn type_tuple(dtype: Dtype) -> #(atom.Atom, Int) {
  #(type_atom(dtype), precision(dtype))
}

pub fn from_type_and_prec(t: atom.Atom, prec: Int) -> Result(Dtype, String) {
  case atom.to_string(t), prec {
    "s", 8 -> Ok(SignedInt8)
    "s", 16 -> Ok(SignedInt16)
    "s", 32 -> Ok(SignedInt32)
    "s", 64 -> Ok(SignedInt64)
    "u", 8 -> Ok(UnsignedInt8)
    "u", 16 -> Ok(UnsignedInt16)
    "u", 32 -> Ok(UnsignedInt32)
    "u", 64 -> Ok(UnsignedInt64)
    "f", 16 -> Ok(Float32)
    "f", 32 -> Ok(Float32)
    "f", 64 -> Ok(Float64)
    t, p ->
      Error(
        "invalid type "
        |> string.append(t)
        |> string.append(" with precision ")
        |> string.append(int.to_string(p)),
      )
  }
}
