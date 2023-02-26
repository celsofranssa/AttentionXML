#!/usr/bin/env bash

python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 0
python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 1
python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 2
python ensemble.py -p resource/result/$2-$1 -t 3
