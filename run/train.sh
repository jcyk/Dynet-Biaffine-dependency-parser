python train.py \
--config_file ../configs/sent.cfg
--in_domain_file ../../sancl_data/gweb-emails-dev.conll \
--model SentParser \
--dev_file ../../sancl_data/gweb-emails-test.conll \
--test_file ../../sancl_data/gweb-emails-test.conll \
--save_dir ../ckpt/mixed-emails \
--load_dir ../ckpt/mixed-emails \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg
--in_domain_file ../../sancl_data/gweb-answers-dev.conll \
--model SentParser \
--dev_file ../../sancl_data/gweb-answers-test.conll \
--test_file ../../sancl_data/gweb-answers-test.conll \
--save_dir ../ckpt/mixed-answers \
--load_dir ../ckpt/mixed-answers \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg
--in_domain_file ../../sancl_data/gweb-newsgroups-dev.conll \
--model SentParser \
--dev_file ../../sancl_data/gweb-newsgroups-test.conll \
--test_file ../../sancl_data/gweb-newsgroups-test.conll \
--save_dir ../ckpt/mixed-newsgroups \
--load_dir ../ckpt/mixed-newsgroups \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg
--in_domain_file ../../sancl_data/gweb-reviews-dev.conll \
--model SentParser \
--dev_file ../../sancl_data/gweb-reviews-test.conll \
--test_file ../../sancl_data/gweb-reviews-test.conll \
--save_dir ../ckpt/mixed-reviews \
--load_dir ../ckpt/mixed-reviews \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg
--in_domain_file ../../sancl_data/gweb-weblogs-dev.conll \
--model SentParser \
--dev_file ../../sancl_data/gweb-weblogs-test.conll \
--test_file ../../sancl_data/gweb-weblogs-test.conll \
--save_dir ../ckpt/mixed-weblogs \
--load_dir ../ckpt/mixed-weblogs \
--dynet-gpu