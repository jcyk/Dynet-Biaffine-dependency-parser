python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../node2vec/ \
--Model NotagParser_auxemb \
--dynet-gpu

python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../node2vec/ \
--Model NotagParser_auxfeat \
--dynet-gpu
