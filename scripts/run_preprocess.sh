
if [ $1 == "EUR-Lex" ]; then
  TRAIN_TEXT="--text-path resource/dataset/$1/train_texts.txt"
  TEST_TEXT="--text-path resource/dataset/$1/test_texts.txt"
else
  TRAIN_TEXT="--text-path resource/dataset/$1/train_raw_texts.txt --tokenized-path resource/dataset/$1/train_texts.txt"
  TEST_TEXT="--text-path resource/dataset/$1/test_raw_texts.txt --tokenized-path resource/dataset/$1/test_texts.txt"
fi

python preprocess.py --dataset $1 --fold $2 $TRAIN_TEXT --label-path resource/dataset/$1/train_labels.txt --vocab-path resource/dataset/$1/vocab.npy --emb-path resource/dataset/$1/emb_init.npy --w2v-model resource/dataset/glove.840B.300d.gensim
python preprocess.py --dataset $1 --fold $2 $TEST_TEXT --label-path resource/dataset/$1/test_labels.txt --vocab-path resource/dataset/$1/vocab.npy

