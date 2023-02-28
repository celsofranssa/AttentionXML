source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

DATA=Eurlex-4k
MODEL=AttentionXML
FOLD=3

bash scripts/run_preprocess.sh $DATA $FOLD
bash scripts/run_xml.sh $DATA $MODEL $FOLD

