-module(kmeans_main).

-export([main/1]).

-define(K, 10).
-define(MAX_ITERATIONS, 20).
-define(TOLERANCE, 1.0e-6).

main(Args) ->
    case Args of
        [DatasetPath] ->
            run(DatasetPath);
        _ ->
            io:format(standard_error, "Usage: _build/default/bin/kmeans <dataset-path>~n", []),
            halt(1)
    end.

run(DatasetPath) ->
    NormalizedPath = filename:absname(DatasetPath),
    case filelib:is_regular(NormalizedPath) of
        true ->
            ok;
        false ->
            fail(["Dataset file does not exist: ", NormalizedPath])
    end,

    info("Starting K-Means baseline with dataset: ~ts~n", [NormalizedPath]),

    case kmeans_csv:load(NormalizedPath) of
        {ok, Dataset} ->
            info(
                "Loaded dataset header with ~B feature columns.~n",
                [kmeans_csv:feature_count(Dataset)]
            ),
            case kmeans:cluster(
                NormalizedPath,
                Dataset,
                ?K,
                ?MAX_ITERATIONS,
                ?TOLERANCE
            ) of
                {ok, Result} ->
                    info(
                        "K-Means finished after ~B iterations.~n",
                        [maps:get(iterations, Result)]
                    ),
                    log_summary(NormalizedPath, Result);
                {error, Reason} ->
                    fail(Reason)
            end;
        {error, Reason} ->
            fail(Reason)
    end.

log_summary(DatasetPath, Result) ->
    io:format("K-Means baseline~n", []),
    io:format("dataset=~ts~n", [DatasetPath]),
    io:format("points=~B~n", [maps:get(point_count, Result)]),
    io:format("k=~B~n", [length(maps:get(centroids, Result))]),
    io:format(
        "features=~ts~n",
        [string:join(maps:get(feature_names, Result), ", ")]
    ),
    io:format("iterations=~B~n", [maps:get(iterations, Result)]),
    io:format("elapsed_ms=~B~n", [maps:get(elapsed_ms, Result)]),
    io:format(
        "sum_squared_error=~ts~n",
        [format_float(maps:get(sum_squared_error, Result))]
    ),
    io:format("clusters:~n", []),
    log_clusters(maps:get(centroids, Result), maps:get(cluster_sizes, Result), 0).

log_clusters([], [], _ClusterIndex) ->
    ok;
log_clusters([Centroid | RestCentroids], [ClusterSize | RestSizes], ClusterIndex) ->
    io:format(
        "  [~B] size=~B centroid=~ts~n",
        [ClusterIndex, ClusterSize, format_centroid(Centroid)]
    ),
    log_clusters(RestCentroids, RestSizes, ClusterIndex + 1).

format_centroid(Centroid) ->
    FormattedValues = [format_float(Value) || Value <- Centroid],
    lists:flatten(["[", string:join(FormattedValues, ", "), "]"]).

format_float(Value) ->
    lists:flatten(io_lib:format("~.6f", [Value])).

info(Format, Args) ->
    io:format(Format, Args).

fail(Reason) ->
    io:format(standard_error, "Error: ~ts~n", [flatten_reason(Reason)]),
    halt(1).

flatten_reason(Reason) when is_binary(Reason) ->
    binary_to_list(Reason);
flatten_reason(Reason) ->
    lists:flatten(io_lib:format("~ts", [Reason])).
