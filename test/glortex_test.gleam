import gleam/option
import gleeunit
import gleeunit/should
import glortex/model
import glortex/tensor

pub fn main() {
  gleeunit.main()
}

// gleeunit test functions end in `_test`
pub fn hello_world_test() {
  1
  |> should.equal(1)
}

// We dont have Nx, like in elixir
// We need some utility for out tests
fn arg_max_rec(idx, max_val, max_idx, vals) -> Int {
  case vals {
    [v0, ..vs] if v0 >. max_val -> arg_max_rec(idx + 1, v0, idx, vs)
    [_, ..vs] -> arg_max_rec(idx + 1, max_val, max_idx, vs)
    [] -> max_idx
  }
}

fn arg_max(vals: List(Float)) -> option.Option(Int) {
  case vals {
    [v0, ..vs] -> option.Some(arg_max_rec(1, v0, 0, vs))
    [] -> option.None
  }
}

// Test the resnet50.onnx model
pub fn resnet50_test() {
  let assert Ok(input) = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  let assert Ok(model) = model.load("./models/resnet50.onnx", [], 3)

  let assert Ok([run_output]) = model.run(model, [input])
  let assert Ok(vals) = tensor.to_float_list(run_output, 10_000_000)
  should.equal(arg_max(vals), option.Some(499))
}
