#source ~/projects/venvs/AttentionXML/bin/activate
#export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/

DATA=Amazon-670k
MODEL=FastAttentionXML
FOLD=0
train_start=$(date)
bash scripts/run_preprocess.sh $DATA $FOLD
bash scripts/run_xml.sh $DATA $MODEL $FOLD
train_end=$(date)

predict_start=$(date)
python ensemble.py -d $1 -m $2 -f $3 -p resource/prediction/$2-$1 -r resource/prediction -t 3
predict_end=$(date)


echo "Training Started at $train_start and ended at $train_end"
echo "Prediction Started at $predict_start and ended at $predict_end"

