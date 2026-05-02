-module(kmeans_csv).

-export([
    load/1,
    feature_count/1,
    feature_names/1,
    read_initial_centroids/3,
    fold_points/4
]).

-define(NON_FEATURE_COLUMNS, ["order_id", "customer_id", "seller_id", "city"]).

load(Path) ->
    with_file(Path, fun(Device) ->
        case io:get_line(Device, "") of
            eof ->
                {error, "CSV file is empty."};
            HeaderLine ->
                case string:trim(trim_line(HeaderLine)) of
                    "" ->
                        {error, "CSV file is empty."};
                    TrimmedHeader ->
                        from_header(TrimmedHeader)
                end
        end
    end).

feature_count(Dataset) ->
    length(maps:get(feature_indexes, Dataset)).

feature_names(Dataset) ->
    maps:get(feature_names, Dataset).

read_initial_centroids(Path, Dataset, K) ->
    with_file(Path, fun(Device) ->
        _ = io:get_line(Device, ""),
        read_initial_centroids(Device, Dataset, K, [])
    end).

fold_points(Path, Dataset, Fun, Acc0) ->
    with_file(Path, fun(Device) ->
        _ = io:get_line(Device, ""),
        fold_points(Device, Dataset, Fun, Acc0, 1)
    end).

from_header(HeaderLine) ->
    Headers = split_csv(HeaderLine),
    IndexedHeaders = lists:zip(Headers, lists:seq(1, length(Headers))),
    {FeatureIndexesRev, FeatureNamesRev} = lists:foldl(
        fun({Header, Index}, {IndexesAcc, NamesAcc}) ->
            TrimmedHeader = string:trim(Header),
            case lists:member(TrimmedHeader, ?NON_FEATURE_COLUMNS) of
                true ->
                    {IndexesAcc, NamesAcc};
                false ->
                    {[Index | IndexesAcc], [TrimmedHeader | NamesAcc]}
            end
        end,
        {[], []},
        IndexedHeaders
    ),

    case FeatureIndexesRev of
        [] ->
            {error, "No numeric feature columns were found in the dataset header."};
        _ ->
            {ok,
                #{
                    headers => Headers,
                    column_count => length(Headers),
                    feature_indexes => lists:reverse(FeatureIndexesRev),
                    feature_names => lists:reverse(FeatureNamesRev)
                }}
    end.

read_initial_centroids(_Device, _Dataset, 0, CentroidsRev) ->
    {ok, lists:reverse(CentroidsRev)};
read_initial_centroids(Device, Dataset, Remaining, CentroidsRev) ->
    case io:get_line(Device, "") of
        eof ->
            {error,
                io_lib:format(
                    "Dataset does not contain enough rows to initialize ~B centroids.",
                    [length(CentroidsRev) + Remaining]
                )};
        Line ->
            case string:trim(trim_line(Line)) of
                "" ->
                    read_initial_centroids(Device, Dataset, Remaining, CentroidsRev);
                _ ->
                    case parse(Line, Dataset) of
                        {ok, Point} ->
                            read_initial_centroids(
                                Device,
                                Dataset,
                                Remaining - 1,
                                [Point | CentroidsRev]
                            );
                        {error, Reason} ->
                            {error, Reason}
                    end
            end
    end.

fold_points(Device, Dataset, Fun, Acc, RowNumber) ->
    case io:get_line(Device, "") of
        eof ->
            {ok, Acc};
        Line ->
            NextRowNumber = RowNumber + 1,
            case string:trim(trim_line(Line)) of
                "" ->
                    fold_points(Device, Dataset, Fun, Acc, NextRowNumber);
                _ ->
                    case parse(Line, Dataset) of
                        {ok, Point} ->
                            case Fun(Point, NextRowNumber, Acc) of
                                {error, _} = Error ->
                                    Error;
                                UpdatedAcc ->
                                    fold_points(
                                        Device,
                                        Dataset,
                                        Fun,
                                        UpdatedAcc,
                                        NextRowNumber
                                    )
                            end;
                        {error, Reason} ->
                            {error,
                                io_lib:format(
                                    "Invalid CSV row at line ~B: ~ts",
                                    [NextRowNumber, Reason]
                                )}
                    end
            end
    end.

parse(Line, Dataset) ->
    Values = split_csv(trim_line(Line)),
    ExpectedColumnCount = maps:get(column_count, Dataset),
    case length(Values) =:= ExpectedColumnCount of
        true ->
            ValueTuple = list_to_tuple(Values),
            parse_features(maps:get(feature_indexes, Dataset), ValueTuple, []);
        false ->
            {error,
                io_lib:format(
                    "Expected ~B columns but found ~B. Row: ~ts",
                    [ExpectedColumnCount, length(Values), trim_line(Line)]
                )}
    end.

parse_features([], _ValueTuple, ParsedRev) ->
    {ok, lists:reverse(ParsedRev)};
parse_features([Index | RestIndexes], ValueTuple, ParsedRev) ->
    case parse_number(element(Index, ValueTuple)) of
        {ok, Value} ->
            parse_features(RestIndexes, ValueTuple, [Value | ParsedRev]);
        {error, Reason} ->
            {error, Reason}
    end.

parse_number(Value0) ->
    Value = string:trim(Value0),
    case string:to_float(Value) of
        {Float, []} ->
            {ok, Float};
        {error, no_float} ->
            case string:to_integer(Value) of
                {Integer, []} ->
                    {ok, Integer * 1.0};
                _ ->
                    {error, io_lib:format("Could not parse numeric value: ~ts", [Value])}
            end;
        _ ->
            {error, io_lib:format("Could not parse numeric value: ~ts", [Value])}
    end.

split_csv(Line) ->
    string:split(Line, ",", all).

trim_line(Line) ->
    string:trim(Line, trailing, "\r\n").

with_file(Path, Fun) ->
    case file:open(Path, [read]) of
        {ok, Device} ->
            try
                Fun(Device)
            after
                file:close(Device)
            end;
        {error, Reason} ->
            {error, file:format_error(Reason)}
    end.
