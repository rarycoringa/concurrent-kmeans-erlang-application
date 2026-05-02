-module(kmeans).

-export([main/1, cluster/5]).

main(Args) ->
    kmeans_main:main(Args).

cluster(DatasetPath, Dataset, K, MaxIterations, Tolerance) ->
    case kmeans_csv:read_initial_centroids(DatasetPath, Dataset, K) of
        {ok, Seeds} ->
            StartTimeMs = erlang:monotonic_time(millisecond),
            io:format(
                "Initialized ~B centroids from the first rows of the dataset.~n",
                [K]
            ),
            case iterate(
                DatasetPath,
                Dataset,
                Seeds,
                kmeans_csv:feature_count(Dataset),
                kmeans_csv:feature_names(Dataset),
                MaxIterations,
                Tolerance,
                1
            ) of
                {ok, Result} ->
                    EndTimeMs = erlang:monotonic_time(millisecond),
                    {ok, Result#{elapsed_ms => EndTimeMs - StartTimeMs}};
                {error, _} = Error ->
                    Error
            end;
        {error, _} = Error ->
            Error
    end.

iterate(
    DatasetPath,
    Dataset,
    Centroids,
    FeatureCount,
    FeatureNames,
    MaxIterations,
    Tolerance,
    Iteration
) when Iteration =< MaxIterations ->
    io:format("Starting iteration ~B of ~B.~n", [Iteration, MaxIterations]),
    InitialSums = zero_matrix(length(Centroids), FeatureCount),
    InitialCounts = lists:duplicate(length(Centroids), 0),
    case kmeans_csv:fold_points(
        DatasetPath,
        Dataset,
        fun(Point, _RowNumber, {SumsAcc, CountsAcc}) ->
            ClosestCentroid = closest_centroid(Point, Centroids),
            {
                add_point(ClosestCentroid, Point, SumsAcc),
                increment_count(ClosestCentroid, CountsAcc)
            }
        end,
        {InitialSums, InitialCounts}
    ) of
        {ok, {Sums, Counts}} ->
            UpdatedCentroids = recompute_centroids(Centroids, Sums, Counts),
            Shift = max_shift(Centroids, UpdatedCentroids),
            io:format(
                "Finished iteration ~B with max centroid shift ~.6f.~n",
                [Iteration, Shift]
            ),
            case Shift =< Tolerance of
                true ->
                    io:format("Converged at iteration ~B.~n", [Iteration]),
                    io:format("Evaluating final clusters.~n", []),
                    evaluate(DatasetPath, Dataset, FeatureNames, UpdatedCentroids, Iteration);
                false ->
                    iterate(
                        DatasetPath,
                        Dataset,
                        UpdatedCentroids,
                        FeatureCount,
                        FeatureNames,
                        MaxIterations,
                        Tolerance,
                        Iteration + 1
                    )
            end;
        {error, _} = Error ->
            Error
    end;
iterate(
    DatasetPath,
    Dataset,
    Centroids,
    _FeatureCount,
    FeatureNames,
    _MaxIterations,
    _Tolerance,
    Iteration
) ->
    io:format("Evaluating final clusters.~n", []),
    evaluate(DatasetPath, Dataset, FeatureNames, Centroids, Iteration - 1).

evaluate(DatasetPath, Dataset, FeatureNames, Centroids, Iterations) ->
    InitialCounts = lists:duplicate(length(Centroids), 0),
    case kmeans_csv:fold_points(
        DatasetPath,
        Dataset,
        fun(Point, _RowNumber, {CountsAcc, SumSquaredErrorAcc, PointCountAcc}) ->
            ClosestCentroid = closest_centroid(Point, Centroids),
            {
                increment_count(ClosestCentroid, CountsAcc),
                SumSquaredErrorAcc + squared_distance(Point, lists:nth(ClosestCentroid, Centroids)),
                PointCountAcc + 1
            }
        end,
        {InitialCounts, 0.0, 0}
    ) of
        {ok, {ClusterSizes, SumSquaredError, PointCount}} ->
            {ok,
                #{
                    centroids => Centroids,
                    feature_names => FeatureNames,
                    cluster_sizes => ClusterSizes,
                    iterations => Iterations,
                    point_count => PointCount,
                    sum_squared_error => SumSquaredError
                }};
        {error, _} = Error ->
            Error
    end.

recompute_centroids(CurrentCentroids, Sums, Counts) ->
    recompute_centroids(CurrentCentroids, Sums, Counts, []).

recompute_centroids([], [], [], UpdatedRev) ->
    lists:reverse(UpdatedRev);
recompute_centroids([Current | RestCurrent], [_Sum | RestSums], [0 | RestCounts], UpdatedRev) ->
    recompute_centroids(RestCurrent, RestSums, RestCounts, [Current | UpdatedRev]);
recompute_centroids([_Current | RestCurrent], [Sum | RestSums], [Count | RestCounts], UpdatedRev) ->
    Updated = [Value / Count || Value <- Sum],
    recompute_centroids(RestCurrent, RestSums, RestCounts, [Updated | UpdatedRev]).

closest_centroid(Point, [FirstCentroid | RestCentroids]) ->
    InitialDistance = squared_distance(Point, FirstCentroid),
    closest_centroid(Point, RestCentroids, 2, 1, InitialDistance).

closest_centroid(_Point, [], _CurrentIndex, BestIndex, _BestDistance) ->
    BestIndex;
closest_centroid(Point, [Centroid | RestCentroids], CurrentIndex, BestIndex, BestDistance) ->
    Distance = squared_distance(Point, Centroid),
    case Distance < BestDistance of
        true ->
            closest_centroid(
                Point,
                RestCentroids,
                CurrentIndex + 1,
                CurrentIndex,
                Distance
            );
        false ->
            closest_centroid(
                Point,
                RestCentroids,
                CurrentIndex + 1,
                BestIndex,
                BestDistance
            )
    end.

squared_distance(Left, Right) ->
    squared_distance(Left, Right, 0.0).

squared_distance([], [], Total) ->
    Total;
squared_distance([LeftValue | RestLeft], [RightValue | RestRight], Total) ->
    Delta = LeftValue - RightValue,
    squared_distance(RestLeft, RestRight, Total + Delta * Delta).

add_point(Index, Point, Sums) ->
    update_nth(Index, fun(Sum) -> add_vectors(Sum, Point) end, Sums).

increment_count(Index, Counts) ->
    update_nth(Index, fun(Count) -> Count + 1 end, Counts).

add_vectors([], []) ->
    [];
add_vectors([Left | RestLeft], [Right | RestRight]) ->
    [Left + Right | add_vectors(RestLeft, RestRight)].

max_shift(CurrentCentroids, UpdatedCentroids) ->
    max_shift(CurrentCentroids, UpdatedCentroids, 0.0).

max_shift([], [], MaxShift) ->
    MaxShift;
max_shift([Current | RestCurrent], [Updated | RestUpdated], MaxShift) ->
    Shift = squared_distance(Current, Updated),
    max_shift(RestCurrent, RestUpdated, erlang:max(MaxShift, Shift)).

zero_matrix(ClusterCount, FeatureCount) ->
    [lists:duplicate(FeatureCount, 0.0) || _ <- lists:seq(1, ClusterCount)].

update_nth(1, Fun, [Value | Rest]) ->
    [Fun(Value) | Rest];
update_nth(Index, Fun, [Value | Rest]) when Index > 1 ->
    [Value | update_nth(Index - 1, Fun, Rest)].
