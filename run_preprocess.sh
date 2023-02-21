### Preprocess
#python preprocess.py \
#--text-path data/Wiki10-31k/train_texts.txt \
#--label-path data/Wiki10-31k/train_labels.txt \
#--vocab-path data/Wiki10-31k/vocab.npy \
#--emb-path data/Wiki10-31k/emb_init.npy \
#--w2v-model data/glove.840B.300d.gensim
#
#python preprocess.py \
#--text-path data/Wiki10-31k/test_texts.txt \
#--label-path data/Wiki10-31k/test_labels.txt \
#--vocab-path data/Wiki10-31k/vocab.npy


# Train and predict as follows:
python main.py \
  --data-cnf configure/datasets/Wiki10-31k.yaml \
  --model-cnf configure/models/AttentionXML-Wiki10-31k.yaml