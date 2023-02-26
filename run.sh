source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/

DATA=Wiki10-31k
MODEL=AttentionXML
FOLD=0

bash scripts/run_preprocess.sh $DATA $FOLD
bash scripts/run_xml.sh $DATA $MODEL

