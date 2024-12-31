% native nif interface in erlang
-module('Elixir.Ortex.Native').

-export([init/3, run/2, from_binary/3, to_binary/3, show_session/1, slice/4, reshape/2,
         concatenate/3, nif_result_to_result/1]).

-nifs([{init, 3},
       {run, 2},
       {from_binary, 3},
       {to_binary, 3},
       {show_session, 1},
       {slice, 4},
       {reshape, 2},
       {concatenate, 3}]).

-on_load load/0.

load() ->
  ok = erlang:load_nif("priv/libortex", 0).

% convert the result type of nif to a result type in gleam
nif_result_to_result({error, Val}) ->
  {error, Val};
nif_result_to_result(Val) ->
  {ok, Val}.

% When loading a NIF module, dummy clauses for all NIF function are required.
% NIF dummies usually just error out when called when the NIF is not loaded, as that should never normally happen.
init(_model_path, _execution_providers, _optimization_level) ->
  exit(nif_library_not_loaded).

run(_model, _inputs) ->
  exit(nif_library_not_loaded).

from_binary(_bin, _shape, _type) ->
  exit(nif_library_not_loaded).

to_binary(_reference, _bits, _limit) ->
  exit(nif_library_not_loaded).

show_session(_model) ->
  exit(nif_library_not_loaded).

slice(_tensor, _start_indicies, _lengths, _strides) ->
  exit(nif_library_not_loaded).

reshape(_tensor, _shape) ->
  exit(nif_library_not_loaded).

concatenate(_tensors_refs, _type, _axis) ->
  exit(nif_library_not_loaded).
