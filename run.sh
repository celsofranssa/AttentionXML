source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

DATA=Wiki10-31k
MODEL=AttentionXML
FOLD=0

#bash scripts/run_preprocess.sh $DATA $FOLD
bash scripts/run_xml.sh $DATA $MODEL

