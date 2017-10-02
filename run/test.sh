python train_notag.py --dynet-gpu --dynet-seed --pretrained_embeddings_file ../../word2vec_emb --save_dir ../ckpt/word2vec

python train_notag.py --dynet-gpu --dynet-seed --pretrained_embeddings_file ../../dep2vec_typed --save_dir ../ckpt/dep2vec_typed

python train_notag.py --dynet-gpu --dynet-seed --pretrained_embeddings_file ../../dep2vec_untyped --save_dir ../ckpt/dep2vec_untyped
