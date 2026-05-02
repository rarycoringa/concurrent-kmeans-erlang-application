clean:
    rebar3 clean

build:
    rebar3 escriptize

run dataset_path:
    rebar3 escriptize && _build/default/bin/kmeans "{{dataset_path}}"
