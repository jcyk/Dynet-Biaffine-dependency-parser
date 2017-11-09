# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle
sys.path.append('..')
import dynet as dy
import numpy as np
import models
from lib import Vocab, DataLoader, MixedDataLoader
from test import test
from config import Configurable
import argparse
if __name__ == "__main__":
	np.random.seed(666)
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/sent.cfg')
	argparser.add_argument('--model', default='DistilltagParser')
	argparser.add_argument('--baseline_path', default='../ckpt/default')
	argparser.add_argument('--critic_scale', type=float, default = 0.01)
	argparser.add_argument('--tag_scale', type=float, default = 0.)
	argparser.add_argument('--ncritic', type=int, default = 5)

	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)
	
	vocab = Vocab(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
	#vocab = cPickle.load(open(os.path.join(args.baseline_path,'vocab')))
	cPickle.dump(vocab, open(config.save_vocab_path, 'w'))
	if args.model == 'DistilltagParser':
		parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.choice_size, randn_init = False)
		#parser.initialize(os.path.join(args.baseline_path,'model'))
		pc = parser.all_parameter_collection
	
	data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
	trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
	
	global_step = 0
	inner_step = 0
	def update_parameters():
		trainer.learning_rate = config.learning_rate*config.decay**(global_step / config.decay_steps)
		trainer.update()

	#parser.load(config.load_model_path)
	best_UAS = 0.
	parser.train_fixed_lstm = True
	parser.set_trainable_flags(train_emb = False, train_lstm = True, train_critic = True, train_score = True)
	while global_step < 50000:
		for words, tags, arcs, rels in data_loader.get_batches(batch_size = config.train_batch_size):
			dy.renew_cg()
			arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs, rels, dep_scale = 1.)
			loss_value = loss.scalar_value()
			loss.backward()
			update_parameters()
			global_step +=1
			if global_step % config.validate_every == 0:
				print '\nTest on development set'
				LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
				print LAS, UAS
				if global_step > config.save_after and UAS > best_UAS:
					best_UAS = UAS
					parser.save(config.save_model_path)
			print arc_accuracy, rel_accuracy, overall_accuracy, loss_value
			
	parser.train_fixed_lstm = False
	parser.set_trainable_flags(train_emb = False, train_lstm = True, train_critic = False, train_score = False)
	global_step = 0
	while global_step < 50000:
		for words, tags, arcs, rels in data_loader.get_batches(batch_size = config.train_batch_size):
			dy.renew_cg()
			loss = parser.run(words, tags, arcs, rels, l2_train = True)
			loss_value = loss.scalar_value()
			loss.backward()
			update_parameters()
			global_step +=1
			if global_step % config.validate_every == 0:
				print '\nTest on development set'
				LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
				print LAS, UAS
				if global_step > config.save_after and UAS > best_UAS:
					best_UAS = UAS
					parser.save(config.save_model_path)
			print loss_value

	global_step = 0
	epoch = 0
	history = lambda x, y : open(os.path.join(config.save_dir, 'valid_history'),'a').write('%.2f %.2f\n'%(x,y))
	while global_step < config.train_iters:
		print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d'%(epoch, )
		epoch += 1
		for words, tags, arcs, rels in data_loader.get_batches(batch_size = config.train_batch_size):
			dy.renew_cg()
			if inner_step % (args.ncritic + 1) == 0 and global_step > 1000:
				parser.set_trainable_flags(train_emb = False, train_lstm = True, train_critic = False, train_score = True)
				arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs, rels, critic_scale = args.critic_scale, dep_scale = 1., tag_scale = args.tag_scale)
			else:
				parser.set_trainable_flags(train_emb = False, train_lstm = False, train_critic = True, train_score = False)
				arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs, rels, critic_scale = args.critic_scale, dep_scale = 0.)	
			loss_value = loss.scalar_value()
			loss.backward()
			sys.stdout.write("Step #%d: Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f, score %.3f\n" %(global_step, arc_accuracy, rel_accuracy, overall_accuracy, loss_value, parser.critic_score))
			sys.stdout.flush()
			update_parameters()
			parser.clip_critic(0.1)
			inner_step += 1
			if inner_step % (args.ncritic + 1) == 0:
				global_step += 1
				if global_step % config.validate_every == 0:
					print '\nTest on development set'
					LAS, UAS = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
					history(LAS, UAS)
					if global_step > config.save_after and UAS > best_UAS:
						best_UAS = UAS
						parser.save(config.save_model_path)
