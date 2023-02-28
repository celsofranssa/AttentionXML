#!/usr/bin/env bash

python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 0
python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 1
python main.py --data-cnf setting/data/$1.yaml --model-cnf setting/model/$2-$1.yaml -t 2
python ensemble.py -d $1 -m $2 -f $3 -p resource/prediction/$2-$1 -r resource/result -t 3


