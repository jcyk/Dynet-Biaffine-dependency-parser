python train_mix_baseline.py --in_domain_file ../../sancl_data/gweb-emails-dev.conll \
							 --dev_file ../../sancl_data/gweb-emails-test.conll \ 
							 --test_file ../../sancl_data/gweb-emails-test.conll \
							 --save_dir ../ckpt/mixed-emails \
							 --load_dir ../ckpt/mixed-emails \
							 --train_iters 30000 \
							 --dynet-gpu

python train_mix_baseline.py --in_domain_file ../../sancl_data/gweb-answers-dev.conll \
							 --dev_file ../../sancl_data/gweb-answers-test.conll \ 
							 --test_file ../../sancl_data/gweb-answers-test.conll \
							 --save_dir ../ckpt/mixed-answers \
							 --load_dir ../ckpt/mixed-answers \
							 --train_iters 30000 \
							 --dynet-gpu

python train_mix_baseline.py --in_domain_file ../../sancl_data/gweb-newsgroups-dev.conll \
							 --dev_file ../../sancl_data/gweb-newsgroups-test.conll \ 
							 --test_file ../../sancl_data/gweb-newsgroups-test.conll \
							 --save_dir ../ckpt/mixed-newsgroups \
							 --load_dir ../ckpt/mixed-newsgroups \
							 --train_iters 30000 \
							 --dynet-gpu

python train_mix_baseline.py --in_domain_file ../../sancl_data/gweb-reviews-dev.conll \
							 --dev_file ../../sancl_data/gweb-reviews-test.conll \ 
							 --test_file ../../sancl_data/gweb-reviews-test.conll \
							 --save_dir ../ckpt/mixed-reviews \
							 --load_dir ../ckpt/mixed-reviews \
							 --train_iters 30000 \
							 --dynet-gpu

python train_mix_baseline.py --in_domain_file ../../sancl_data/gweb-weblogs-dev.conll \
							 --dev_file ../../sancl_data/gweb-weblogs-test.conll \ 
							 --test_file ../../sancl_data/gweb-weblogs-test.conll \
							 --save_dir ../ckpt/mixed-weblogs \
							 --load_dir ../ckpt/mixed-weblogs \
							 --train_iters 30000 \
							 --dynet-gpu