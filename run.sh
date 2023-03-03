source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/

DATA=Amazon-670k
MODEL=FastAttentionXML
FOLD=0

#bash scripts/run_preprocess.sh $DATA $FOLD
bash scripts/run_xml.sh $DATA $MODEL $FOLD

