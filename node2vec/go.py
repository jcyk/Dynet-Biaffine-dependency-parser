import os

pq_value = [0.25, 0.5, 1, 2, 4]

for p in pq_value:
	for q in pq_value:
		for walk_length in [10]:
			for window_size in [5]:
				os.system("python main.py --input result0 result1 result2 result3 ../../sancl_data/wsj_norm_train.conll --output emb_p_%.2f_q_%.2f_walk_length_%d_window_size_%d --walk-length %d --iter 10 --p %.2f --q %.2f --window-size %d --weighted --directed"%(p, q, walk_length, window_size, walk_length,p, q, window_size))