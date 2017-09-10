python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../../node2vec/emb_p_0.25_q_0.25_walk_length_10_window_size_5 \
--Model NotagParser_auxemb \
--save_dir emb_p_0.25_q_0.25_walk_length_10_window_size_5_auxemb \
--dynet-gpu

python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../node2vec/emb_p_0.25_q_0.25_walk_length_10_window_size_5 \
--Model NotagParser_auxfeat \
--save_dir emb_p_0.25_q_0.25_walk_length_10_window_size_5_auxfeat \
--dynet-gpu


python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../../node2vec/emb_p_0.25_q_0.50_walk_length_10_window_size_5 \
--Model NotagParser_auxemb \
--save_dir emb_p_0.25_q_0.50_walk_length_10_window_size_5_auxemb \
--dynet-gpu

python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../node2vec/emb_p_0.25_q_0.50_walk_length_10_window_size_5 \
--Model NotagParser_auxfeat \
--save_dir emb_p_0.25_q_0.50_walk_length_10_window_size_5_auxfeat \
--dynet-gpu