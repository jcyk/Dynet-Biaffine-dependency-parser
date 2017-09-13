import os


cmd = """
python train_notag_aux.py \
--config_file ../configs/aux.cfg \
--aux_pretrained_embeddings_file ../../node2vec/%s \
--model NotagParser_auxemb \
--save_dir ../ckpt/%s \
--dynet-gpu
"""

print cmd

pq_value = [0.25, 0.5, 1, 2, 4]


for p in pq_value:
	for q in pq_value:
		for walk_length in [10]:
			for window_size in [5]:
				fname = "emb_p_%.2f_q_%.2f_walk_length_%d_window_size_%d"%(p,q,walk_length,window_size)
				os.system(cmd%(fname, fname+'_auxemb'))
				