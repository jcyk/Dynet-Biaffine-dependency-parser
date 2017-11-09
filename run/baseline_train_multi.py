# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle
sys.path.append('..')
import dynet as dy
import numpy as np
import models
from lib import Vocab, DataLoader
from test import test
from config import Configurable

import argparse

class MixedDataLoader(object):
	def __init__(self, files, ratios, n_bkts, vocabs):
		assert isinstance(files, list)
		assert isinstance(ratios, list)
		test = DataLoader(files[0], n_bkts, vocabs[0])
		bucket_sizes = test._bucket_sizes
		del test
		self._loaders = [ DataLoader(_file, n_bkts, vocab, bucket_sizes = bucket_sizes) for _file, vocab in zip(files,vocabs)]
		self._ratios = ratios

	def get_batches(self, batch_size):
		generators = [loader.get_batches(int(ratio * batch_size)) for loader, ratio in zip(self._loaders, self._ratios)]
		for batch_tuple in zip(*generators):
			yield batch_tuple

if __name__ == "__main__":
	np.random.seed(666)
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/default.cfg')
	argparser.add_argument('--out_domain_file', default='../../sancl_data/gweb-emails-dev.conll')
	argparser.add_argument('--model', default='BaseParser')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)

	vocab = Vocab(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
	vocab1 = Vocab(config.out_domain_file, config.pretrained_embeddings_file, config.min_occur_count)
	vocab.merge_with(vocab1)
	vocab0 = Vocab(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
	vocab0.merge_with(vocab1, only_words = True)
	vocab1.merge_with(vocab0, only_words = True)
	cPickle.dump(vocab, open(config.save_vocab_path, 'w'))
	cPickle.dump(vocab0, open(config.save_vocab_path+'0', 'w'))
	cPickle.dump(vocab1, open(config.save_vocab_path+'1', 'w'))

	parser = Parser(vocab, vocab1, vocab2, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp)
	data_loader = MixedDataLoader([config.train_file, args.in_domain_file], [0.5, 0.5], config.num_buckets_train, [vocab0, vocab1])
	pc = parser.parameter_collection
	trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
	
	global_step = 0
	def update_parameters():
		trainer.learning_rate = config.learning_rate*config.decay**(global_step / config.decay_steps)
		trainer.update()

	epoch = 0
	best_UAS = 0.
	history = lambda x, y : open(os.path.join(config.save_dir, 'valid_history'),'a').write('%.2f %.2f\n'%(x,y))
	while global_step < config.train_iters:
		print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d'%(epoch, )
		epoch += 1
		for _in, _out in data_loader.get_batches(batch_size = config.train_batch_size):
			for domain, _inputs in enumerate([_in, _out]):
				words, tags, arcs, rels = _inputs
				dy.renew_cg()
				arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs, rels, data_type = domain)
				loss_value = loss.scalar_value()
				loss.backward()
				update_parameters()
				sys.stdout.write("Step #%d: Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f\r\r" %(global_step, arc_accuracy, rel_accuracy, overall_accuracy, loss_value))
				sys.stdout.flush()

				global_step += 1
				if global_step % config.validate_every == 0:
					print '\nTest on development set'
					LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
					history(LAS, UAS)
					if global_step > config.save_after and UAS > best_UAS:
						best_UAS = UAS
						parser.save(config.save_model_path)