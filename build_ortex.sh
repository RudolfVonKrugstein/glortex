#!/usr/bin/env bash

set -e

pushd deps

if [ -d ortex ]; then
  cd ortex
  git pull
else
  git clone https://github.com/elixir-nx/ortex.git
  cd ortex
fi

cd native/ortex
cargo build --release
cp target/release/libortex.so ../../../../priv

popd
