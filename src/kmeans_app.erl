%%%-------------------------------------------------------------------
%% @doc kmeans public API
%% @end
%%%-------------------------------------------------------------------

-module(kmeans_app).

-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
    kmeans_sup:start_link().

stop(_State) ->
    ok.

%% internal functions
