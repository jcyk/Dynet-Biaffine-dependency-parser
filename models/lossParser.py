# -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np

from lib import uniLSTM, leaky_relu, bilinear, orthonormal_initializer, arc_argmax, rel_argmax, orthonormal_VanillaLSTMBuilder

class LossParser(object):
	def __init__(self, vocab,
					   word_dims,
					   tag_dims,
					   dropout_emb,
					   lstm_layers,
					   lstm_hiddens,
					   dropout_lstm_input,
					   dropout_lstm_hidden,
					   mlp_arc_size,
					   mlp_rel_size,
					   dropout_mlp,
					   choice_size,
					   randn_init = False
					   ):

		all_params = dy.ParameterCollection()
		pc = all_params.add_subcollection()
		trainable_params = all_params.add_subcollection()

		self._vocab = vocab
		self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
		self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs())

		self.dropout_emb = dropout_emb
		self.LSTM_builders = []
		f = orthonormal_VanillaLSTMBuilder(lstm_layers, word_dims, lstm_hiddens, pc, randn_init)
		b = orthonormal_VanillaLSTMBuilder(lstm_layers, word_dims, lstm_hiddens, pc, randn_init)
		self.LSTM_builders.append((f,b))
		self.dropout_lstm_input = dropout_lstm_input
		self.dropout_lstm_hidden = dropout_lstm_hidden

		mlp_size = mlp_arc_size+mlp_rel_size
		W = orthonormal_initializer(mlp_size, 2*lstm_hiddens, randn_init)
		self.mlp_dep_W = pc.parameters_from_numpy(W)
		self.mlp_head_W = pc.parameters_from_numpy(W)
		self.mlp_dep_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_head_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_arc_size = mlp_arc_size
		self.mlp_rel_size = mlp_rel_size
		self.dropout_mlp = dropout_mlp

		self.arc_W = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init = dy.ConstInitializer(0.))
		self.rel_W = pc.add_parameters((vocab.rel_size*(mlp_rel_size +1) , mlp_rel_size + 1), init = dy.ConstInitializer(0.))

		self.att_W = trainable_params.parameters_from_numpy(orthonormal_initializer(1, 2*lstm_hiddens))
		self.choice_W = trainable_params.parameters_from_numpy(orthonormal_initializer(choice_size, 2*lstm_hiddens))
		self.choice_b = trainable_params.add_parameters((choice_size,), init = dy.ConstInitializer(0.))
		self.judge_W = trainable_params.add_parameters((1, choice_size), init = dy.ConstInitializer(0.))
		self.judge_b = trainable_params.add_parameters((1,), init = dy.ConstInitializer(0.))

		self._critic_params = [self.att_W, self.choice_W, self.choice_b, self.judge_W, self.judge_b]

		self.tgt_embs_W = trainable_params.add_parameters((vocab.words_in_train, lstm_hiddens))
		self.tgt_embs_b = trainable_params.add_parameters((vocab.words_in_train,), init = dy.ConstInitializer(0.))
		self.tag_embs_W = trainable_params.add_parameters((vocab.tag_size, 2*lstm_hiddens))
		self.tag_embs_b = trainable_params.add_parameters(vocab.tag_size, init = dy.ConstInitializer(0.))

		self._all_params = all_params
		self._pc = pc
		self._trainable_params = trainable_params
		self.set_trainable_flags(False, False, False, False)

	def set_trainable_flags(self, train_emb, train_lstm, train_critic, train_score):
		self.train_emb = train_emb
		self.train_lstm = train_lstm
		self.train_critic = train_critic
		self.train_score = train_score

	@property
	def all_parameter_collection(self):
		return self._all_params

	@property 
	def trainable_parameter_collection(self):
		return self._trainable_params

	def run(self, word_inputs, tag_inputs = None, arc_targets = None, rel_targets = None, isTrain = True, critic_scale =0., dep_scale = 0., lm_scale= 0., tag_scale = 0.): 
		# inputs, targets: seq_len x batch_size
		def dynet_flatten_numpy(ndarray):
			return np.reshape(ndarray, (-1,), 'F')

		batch_size = word_inputs.shape[1]
		seq_len = word_inputs.shape[0]
		mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
		num_tokens = int(np.sum(mask))
		
		if isTrain or arc_targets is not None:
			mask_1D = dynet_flatten_numpy(mask)
			mask_1D_tensor = dy.inputTensor(mask_1D, batched = True)
		
		word_embs = [dy.lookup_batch(self.word_embs, np.where( w<self._vocab.words_in_train, w, self._vocab.UNK), update= self.train_emb) + dy.lookup_batch(self.pret_word_embs, w, update = False) for w in word_inputs]
		
		if isTrain:
			word_embs= [ dy.dropout_dim(w, 0, self.dropout_emb) for w in word_embs]
		
		f_recur = uniLSTM(self.LSTM_builders[0][0], word_embs, batch_size, self.dropout_lstm_input if isTrain else 0., self.dropout_lstm_hidden if isTrain else 0., update = self.train_lstm)
		b_recur = uniLSTM(self.LSTM_builders[0][1], word_embs[::-1], batch_size, self.dropout_lstm_input if isTrain else 0., self.dropout_lstm_hidden if isTrain else 0., update = self.train_lstm)
		b_recur = b_recur[::-1]
		fb_recur = [dy.concatenate([f,b]) for f, b in zip(f_recur, b_recur)]
		top_recur = dy.concatenate_cols(fb_recur)		
		if lm_scale != 0.:
			W_tgt, b_tgt = dy.parameter(self.tgt_embs_W), dy.parameter(self.tgt_embs_b)
			losses = []
			for h, tgt, msk in zip(f_recur[:-1], word_inputs[1:], mask[1:]):
				y  = W_tgt * h + b_tgt
				tgt = np.where( tgt<self._vocab.words_in_train, tgt, self._vocab.UNK)
				losses.append(dy.pickneglogsoftmax_batch(y, tgt) * dy.inputTensor(msk, batched=True))
			
			for h, tgt, msk in zip(b_recur[1:], word_inputs[:-1], mask[:-1]):
				y  = W_tgt * h + b_tgt
				tgt = np.where( tgt<self._vocab.words_in_train, tgt, self._vocab.UNK)
				losses.append(dy.pickneglogsoftmax_batch(y, tgt) * dy.inputTensor(msk, batched=True))
			lm_loss = dy.sum_batches(dy.esum(losses))/ num_tokens
			self.lm_loss  = lm_loss.scalar_value()
		if tag_scale != 0.:
			W_tag, b_tag = dy.parameter(self.tag_embs_W), dy.parameter(self.tag_embs_b)
			losses = []
			for h, tgt, msk in zip(fb_recur, tag_inputs, mask):
				y = W_tag * h + b_tag
				losses.append(dy.pickneglogsoftmax_batch(y, tgt) * dy.inputTensor(msk, batched = True))
			tag_loss = dy.sum_batches(dy.esum(losses)) / num_tokens
			self.tag_loss = tag_loss.scalar_value()

		if critic_scale != 0.:
			W_att = dy.parameter(self.att_W, update = self.train_critic)
			att_weights = dy.softmax(dy.reshape(W_att * top_recur,(seq_len,), batch_size))
			W_choice, b_choice = dy.parameter(self.choice_W, update = self.train_critic), dy.parameter(self.choice_b, update = self.train_critic)
			W_judge, b_judge = dy.parameter(self.judge_W, update = self.train_critic), dy.parameter(self.judge_b, update = self.train_critic)
			choice_logits = leaky_relu(dy.affine_transform([b_choice, W_choice, top_recur])) * att_weights
			in_decisions = dy.affine_transform([b_judge, W_judge, choice_logits])
			critic_loss = dy.sum_batches(in_decisions) / batch_size
			self.critic_loss = critic_loss.scalar_value()

		if isTrain:
			top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

		W_dep, b_dep = dy.parameter(self.mlp_dep_W, update = self.train_score), dy.parameter(self.mlp_dep_b, update = self.train_score)
		W_head, b_head = dy.parameter(self.mlp_head_W, update = self.train_score), dy.parameter(self.mlp_head_b, update = self.train_score)
		dep, head = leaky_relu(dy.affine_transform([b_dep, W_dep, top_recur])),leaky_relu(dy.affine_transform([b_head, W_head, top_recur]))
		if isTrain:
			dep, head= dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)
		
		dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
		head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

		W_arc = dy.parameter(self.arc_W, update = self.train_score)
		arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs= 1, bias_x = True, bias_y = False)
		# (#head x #dep) x batch_size
		
		flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len * batch_size)
		# (#head ) x (#dep x batch_size)

		arc_preds = arc_logits.npvalue().argmax(0)
		# seq_len x batch_size

		if isTrain or arc_targets is not None:
			arc_correct = np.equal(arc_preds, arc_targets).astype(np.float32) * mask
			arc_accuracy = np.sum(arc_correct) / num_tokens
			targets_1D = dynet_flatten_numpy(arc_targets)
			losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)
			arc_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
		
		if not isTrain:
			arc_probs = np.transpose(np.reshape(dy.softmax(flat_arc_logits).npvalue(), (seq_len, seq_len, batch_size), 'F'))
			# #batch_size x #dep x #head

		W_rel = dy.parameter(self.rel_W, update = self.train_score)
		rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size, num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# (#head x rel_size x #dep) x batch_size
		
		flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
		# (#head x rel_size) x (#dep x batch_size)

		partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D if isTrain else dynet_flatten_numpy(arc_preds))
		# (rel_size) x (#dep x batch_size)

		if isTrain or arc_targets is not None:
			rel_preds = partial_rel_logits.npvalue().argmax(0)
			targets_1D = dynet_flatten_numpy(rel_targets)
			rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
			rel_accuracy = np.sum(rel_correct)/ num_tokens
			losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
			rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
		
		if not isTrain:
			rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), (self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
			# batch_size x #dep x #head x #nclasses
	
		if isTrain or arc_targets is not None:
			dep_loss = arc_loss + rel_loss
			self.dep_loss = dep_loss.scalar_value()
			loss = (dep_scale * dep_loss if dep_scale != 0. else 0. ) + ( critic_scale * critic_loss  if critic_scale != 0. else 0.) + ( lm_scale * lm_loss  if lm_scale != 0. else 0.)  + ( tag_scale * tag_loss  if tag_scale != 0. else 0.) 
			correct = rel_correct * dynet_flatten_numpy(arc_correct)
			overall_accuracy = np.sum(correct) / num_tokens 
		
		if isTrain:
			return arc_accuracy, rel_accuracy, overall_accuracy, loss
		
		outputs = []
		
		for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
			# parse sentences one by one
			msk[0] = 1.
			sent_len = int(np.sum(msk))
			arc_pred = arc_argmax(arc_prob, sent_len, msk)
			rel_prob = rel_prob[np.arange(len(arc_pred)),arc_pred]
			rel_pred = rel_argmax(rel_prob, sent_len)
			outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))
		
		if arc_targets is not None:
			return arc_accuracy, rel_accuracy, overall_accuracy, outputs
		return outputs

	def initialize(self, baseline_params):
		self._pc.populate(baseline_params)
		
	def clip_critic(self, range_value):
		for param in self._critic_params:
			param.clip_inplace(-range_value, range_value)
	def save(self, save_path):
		self._all_params.save(save_path)
	def load(self, load_path):
		self._all_params.populate(load_path)
