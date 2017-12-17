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
	def __init__(self, files, ratios, n_bkts, vocab):
		assert isinstance(files, list)
		assert isinstance(ratios, list)
		test = DataLoader(files[0], n_bkts, vocab.vocabs[0])
		bucket_sizes = test._bucket_sizes
		del test
		self._loaders = [ DataLoader(_file, n_bkts, vocab, bucket_sizes = bucket_sizes) for _file, vocab in zip(files,vocab.vocabs)]
		self._ratios = ratios

	def get_batches(self, batch_size):
		generators = [loader.get_batches(int(ratio * batch_size)) for loader, ratio in zip(self._loaders, self._ratios)]
		for batch_tuple in zip(*generators):
			yield batch_tuple

class MixedVocab(object):
	PAD, ROOT, UNK = 0, 1, 2
	def __init__ (self, vocabs):
		self.vocabs = vocabs
		self.get_pret_embs = self.vocabs[0].get_pret_embs
		self.get_word_embs = self.vocabs[0].get_word_embs
		self.word2id = self.vocabs[0].word2id
		self.id2word = self.vocabs[0].id2word
		self.rel2id = [ vocab.rel2id for vocab in self.vocabs]
		self.id2rel = [ vocab.id2rel for vocab in self.vocabs]
		self.tag2id = [ vocab.tag2id for vocab in self.vocabs]
		
	@property 
	def words_in_train(self):
		return self.vocabs[0]._words_in_train_data

	@property
	def vocab_size(self):
		return len(self.vocabs[0]._id2word)

	@property
	def tag_size(self):
		return [len(vocab._id2tag) for vocab in self.vocabs]

	@property
	def rel_size(self):
		return [len(vocab._id2rel) for vocab in self.vocabs]

if __name__ == "__main__":
	np.random.seed(666)
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/multi.cfg')
	argparser.add_argument('--out_domain_file', default='../../multi/aaai.conll')
	argparser.add_argument('--model', default='BaseParserMulti')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args, model = args.model)
	Parser = getattr(models, args.model)

	vocab0 = Vocab(config.train_file, None, config.min_occur_count)
	vocab1 = Vocab(args.out_domain_file, None, config.min_occur_count)
	Vocab.merge(vocab0, vocab1, config.pretrained_embeddings_file)
	vocab = MixedVocab([vocab0, vocab1])
	cPickle.dump(vocab0, open(config.save_vocab_path+"0", 'w'))
	cPickle.dump(vocab1, open(config.save_vocab_path+"1", 'w'))

	parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp)
	data_loader = MixedDataLoader([config.train_file, args.out_domain_file], [1., 1.], config.num_buckets_train, vocab)
	pc = parser.parameter_collection
	parser_trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
	tagger_trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
	global_step = 0

	def update_parser_parameters():
		parser_trainer.learning_rate = config.learning_rate*config.decay**(global_step / config.decay_steps)
		parser_trainer.update()

	def update_tagger_parameters():
		tagger_trainer.learning_rate = config.learning_rate*config.decay**(global_step / config.decay_steps)
		tagger_trainer.update()

	epoch = 0
	best_UAS = 0.
	history = lambda x, y : open(os.path.join(config.save_dir, 'valid_history'),'a').write('%.2f %.2f\n'%(x,y))
	while global_step < config.train_iters:
		print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d'%(epoch, )
		epoch += 1
		for _in, _out in data_loader.get_batches(batch_size = config.train_batch_size):
			for domain, _inputs in enumerate([_in, _out]):
				if domain == 1:
					continue
				words, tags, arcs, rels = _inputs
				dy.renew_cg()
				if epoch<=5 or (5<epoch <= 50 and global_step % 2 == 0) or (50<epoch and global_step % 3 == 0):
					tag_acc, loss = parser.run(words, tags, arcs, rels, data_type = domain, tag_turn = True)
					loss_value = loss.scalar_value()
					loss.backward()
					update_tagger_parameters()
					print "Step #%d: Domain: %d Acc: tag %.2f loss %.3f" %(global_step, domain,tag_acc, loss_value)
				else:
					arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs, rels, data_type = domain)
					loss_value = loss.scalar_value()
					print "Step #%d: Domain: %d Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f" %(global_step, domain, arc_accuracy, rel_accuracy, overall_accuracy, loss_value)
					loss.backward()
					update_parser_parameters()
					
			global_step += 1
			if global_step % config.validate_every == 0:
				print '\nTest on development set'
				LAS, UAS = test(parser, vocab.vocabs[0], config.num_buckets_valid, config.test_batch_size, config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
				history(LAS, UAS)
				if global_step > config.save_after and UAS > best_UAS:
					best_UAS = UAS
					parser.save(config.save_model_path)