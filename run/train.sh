python baseline_train3000.py \
--save_dir ../ckpt/sota \
--dynet-gpu

python baseline_train3000.py \
--train_file ../../sancl_data/gweb-emails-dev.conll \
--dev_file ../../sancl_data/gweb-emails-test.conll \
--test_file ../../sancl_data/gweb-emails-test.conll \
--save_dir ../ckpt/self-emails \
--num_buckets_train 10 \
--train_iters 30000 \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg \
--in_domain_file ../../sancl_data/gweb-emails-dev.conll \
--model SentParser \
--domain_loss_scale 0. \
--dev_file ../../sancl_data/gweb-emails-test.conll \
--test_file ../../sancl_data/gweb-emails-test.conll \
--save_dir ../ckpt/mixed-emails-0. \
--dynet-gpu

python train.py \
--config_file ../configs/sent.cfg \
--in_domain_file ../../sancl_data/gweb-emails-dev.conll \
--model SentParser \
--domain_loss_scale 0.01 \
--dev_file ../../sancl_data/gweb-emails-test.conll \
--test_file ../../sancl_data/gweb-emails-test.conll \
--save_dir ../ckpt/mixed-emails-0.01 \
--dynet-gpu

#python train.py \
#--config_file ../configs/sent.cfg \
#--in_domain_file ../../sancl_data/gweb-emails-dev.conll \
#--model SentParser \
#--domain_loss_scale 0. \
#--dev_file ../../sancl_data/gweb-emails-test.conll \
#--test_file ../../sancl_data/gweb-emails-test.conll \
#--save_dir ../ckpt/mixed-emails-0. \
#--dynet-gpu

#python train.py \
#--config_file ../configs/sent.cfg \
#--in_domain_file ../../sancl_data/gweb-answers-dev.conll \
#--model SentParser \
#--dev_file ../../sancl_data/gweb-answers-test.conll \
#--test_file ../../sancl_data/gweb-answers-test.conll \
#--save_dir ../ckpt/mixed-answers \
#--load_dir ../ckpt/mixed-answers \
#--dynet-gpu

#python train.py \
#--config_file ../configs/sent.cfg \
#--in_domain_file ../../sancl_data/gweb-weblogs-dev.conll \
#--model SentParser \
#--dev_file ../../sancl_data/gweb-weblogs-test.conll \
#--test_file ../../sancl_data/gweb-weblogs-test.conll \
#--save_dir ../ckpt/mixed-weblogs \
#--load_dir ../ckpt/mixed-weblogs \
#--dynet-gpu

#python train.py \
#--config_file ../configs/sent.cfg \
#--in_domain_file ../../sancl_data/gweb-newsgroups-dev.conll \
#--model SentParser \
#--dev_file ../../sancl_data/gweb-newsgroups-test.conll \
#--test_file ../../sancl_data/gweb-newsgroups-test.conll \
#--save_dir ../ckpt/mixed-newsgroups \
#--load_dir ../ckpt/mixed-newsgroups \
#--dynet-gpu

#python train.py \
#--config_file ../configs/sent.cfg \
#--in_domain_file ../../sancl_data/gweb-reviews-dev.conll \
#--model SentParser \
#--dev_file ../../sancl_data/gweb-reviews-test.conll \
#--test_file ../../sancl_data/gweb-reviews-test.conll \
#--save_dir ../ckpt/mixed-reviews \
#--load_dir ../ckpt/mixed-reviews \
#--dynet-gpu