source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/

DATA=Eurlex-4k
MODEL=AttentionXML
FOLD=0

time_start=$(date '+%Y-%m-%d %H:%M:%S')
bash scripts/run_preprocess.sh $DATA $FOLD
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > preprocess_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
bash scripts/run_xml.sh $DATA $MODEL $FOLD
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > fit_time.txt

time_start=$(date '+%Y-%m-%d %H:%M:%S')
python ensemble.py -d $DATA -m $MODEL -f $FOLD -p resource/prediction/$MODEL-$DATA -r resource/prediction -t 3
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > predict.txt

