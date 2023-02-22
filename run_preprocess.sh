# activate venv and set Python path
source ~/projects/venvs/AttentionXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/AttentionXML/
## Preprocess
python preprocess.py \
--text-path data/Wiki10-31k/train_texts.txt \
--label-path data/Wiki10-31k/train_labels.txt \
--vocab-path data/Wiki10-31k/vocab.npy \
--emb-path data/Wiki10-31k/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

python preprocess.py \
--text-path data/Wiki10-31k/test_texts.txt \
--label-path data/Wiki10-31k/test_labels.txt \
--vocab-path data/Wiki10-31k/vocab.npy
