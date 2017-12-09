# -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np

from lib import biLSTM, leaky_relu, bilinear, orthonormal_initializer, arc_argmax, rel_argmax, orthonormal_VanillaLSTMBuilder

class BaseParserMulti(object):
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
					   randn_init = False
					   ):
		pc = dy.ParameterCollection()
		
		self._vocab = vocab
		self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
		self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs())
		self.dropout_emb = dropout_emb
		self.LSTM_builders = []
		f = orthonormal_VanillaLSTMBuilder(1, word_dims+2*tag_dims, lstm_hiddens, pc, randn_init)
		b = orthonormal_VanillaLSTMBuilder(1, word_dims+2*tag_dims, lstm_hiddens, pc, randn_init)
		self.LSTM_builders.append((f,b))
		for i in xrange(lstm_layers-1):
			f = orthonormal_VanillaLSTMBuilder(1, 2*lstm_hiddens, lstm_hiddens, pc, randn_init)
			b = orthonormal_VanillaLSTMBuilder(1, 2*lstm_hiddens, lstm_hiddens, pc, randn_init)
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

		self.arc_W0 = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init = dy.ConstInitializer(0.))
		self.arc_W1 = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init = dy.ConstInitializer(0.)) 
		self.arc_Ws = [self.arc_W0, self.arc_W1]
		self.rel_W0 = pc.add_parameters((vocab.rel_size[0]*(mlp_rel_size +1) , mlp_rel_size + 1), init = dy.ConstInitializer(0.))
		self.rel_W1 = pc.add_parameters((vocab.rel_size[1]*(mlp_rel_size +1) , mlp_rel_size + 1), init = dy.ConstInitializer(0.))
		self.rel_Ws = [self.rel_W0, self.rel_W1]
 		
 		f = orthonormal_VanillaLSTMBuilder(1, word_dims, tag_dims//2, pc, randn_init)
		b = orthonormal_VanillaLSTMBuilder(1, word_dims, tag_dims//2, pc, randn_init)
		self.tag_LSTM_builders0 = [(f,b)]

		f = orthonormal_VanillaLSTMBuilder(1, word_dims, tag_dims//2, pc, randn_init)
		b = orthonormal_VanillaLSTMBuilder(1, word_dims, tag_dims//2, pc, randn_init)
		self.tag_LSTM_builders1 = [(f,b)]

		self.tag_embs_W0 = pc.add_parameters((vocab.tag_size[0], tag_dims))
		self.tag_embs_b0 = pc.add_parameters(vocab.tag_size[0], init = dy.ConstInitializer(0.))
		self.tag_embs_W1 = pc.add_parameters((vocab.tag_size[1], tag_dims))
		self.tag_embs_b1 = pc.add_parameters(vocab.tag_size[1], init = dy.ConstInitializer(0.))
		self.tag_embs_Ws = [self.tag_embs_W0, self.tag_embs_W1]
		self.tag_embs_bs = [self.tag_embs_b0, self.tag_embs_b1]
		self._pc = pc
		
		def _emb_mask_generator(seq_len, batch_size):
			ret = []
			for i in xrange(seq_len):
				word_mask = np.random.binomial(1, 1. - dropout_emb, batch_size).astype(np.float32)
				tag_mask = np.random.binomial(1, 1. - dropout_emb, batch_size).astype(np.float32)
				scale = 3. / (2.*word_mask + tag_mask + 1e-12)
				word_mask *= scale
				tag_mask *= scale
				word_mask = dy.inputTensor(word_mask, batched = True)
				tag_mask = dy.inputTensor(tag_mask, batched = True)
				ret.append((word_mask, tag_mask))
			return ret
		self.generate_emb_mask = _emb_mask_generator
	@property 
	def parameter_collection(self):
		return self._pc

	def run(self, word_inputs, tag_inputs, arc_targets = None, rel_targets = None, isTrain = True, data_type = 0, tag_turn = False):
		# inputs, targets: seq_len x batch_size
		self.rel_W = self.rel_Ws[data_type]
		rel_size = self._vocab.rel_size[data_type]
		def dynet_flatten_numpy(ndarray):
			return np.reshape(ndarray, (-1,), 'F')

		batch_size = word_inputs.shape[1]
		seq_len = word_inputs.shape[0]
		mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
		num_tokens = int(np.sum(mask))
		
		if isTrain or arc_targets is not None:
			mask_1D = dynet_flatten_numpy(mask)
			mask_1D_tensor = dy.inputTensor(mask_1D, batched = True)
		
		word_embs = [dy.lookup_batch(self.word_embs, np.where( w<self._vocab.words_in_train, w, self._vocab.UNK)) + dy.lookup_batch(self.pret_word_embs, w, update = False) for w in word_inputs]
		
		if isTrain and tag_turn:
			word_embs= [ dy.dropout_dim(w, 0, self.dropout_emb) for w in word_embs]

		tag_recur0 = biLSTM(self.tag_LSTM_builders0, word_embs, batch_size, self.dropout_lstm_input if (isTrain and tag_turn) else 0., self.dropout_lstm_hidden if (isTrain and tag_turn) else 0.)
		tag_recur1 = biLSTM(self.tag_LSTM_builders1, word_embs, batch_size, self.dropout_lstm_input if (isTrain and tag_turn) else 0., self.dropout_lstm_hidden if (isTrain and tag_turn) else 0.)
		
		W_tag, b_tag = dy.parameter(self.tag_embs_Ws[data_type]), dy.parameter(self.tag_embs_bs[data_type])
		tag_recur = (tag_recur0 if data_type == 0 else tag_recur1)
		losses = []

		if tag_turn or not isTrain:
			correct = 0
			for h, tgt, msk in zip(tag_recur, tag_inputs, mask):
				y = W_tag * h + b_tag
				losses.append(dy.pickneglogsoftmax_batch(y, tgt) * dy.inputTensor(msk, batched = True))
				correct += np.sum(np.equal(y.npvalue().argmax(0), tgt).astype(np.float32) * msk)
			tag_acc = correct / num_tokens
			if not isTrain:
				print tag_acc, num_tokens
			if isTrain:
				tag_loss = dy.sum_batches(dy.esum(losses)) / num_tokens
				return tag_acc, tag_loss
		emb_masks = self.generate_emb_mask(seq_len, batch_size)
		emb_inputs = [ dy.concatenate([dy.cmult(w, wm), dy.cmult(pos0,posm), dy.cmult(pos1,posm)]) for w, pos0, pos1, (wm, posm) in zip(word_embs,tag_recur0, tag_recur1, emb_masks)]
		top_recur = dy.concatenate_cols(biLSTM(self.LSTM_builders, emb_inputs, batch_size, self.dropout_lstm_input if isTrain else 0., self.dropout_lstm_hidden if isTrain else 0.))
		if isTrain:
			top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

		W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
		W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
		dep, head = leaky_relu(dy.affine_transform([b_dep, W_dep, top_recur])),leaky_relu(dy.affine_transform([b_head, W_head, top_recur]))
		if isTrain:
			dep, head= dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)
		
		dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
		head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

		W_arc = dy.parameter(self.arc_Ws[data_type])
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

		W_rel = dy.parameter(self.rel_W)

		rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size, num_outputs = rel_size, bias_x = True, bias_y = True)
		# (#head x rel_size x #dep) x batch_size
		
		flat_rel_logits = dy.reshape(rel_logits, (seq_len, rel_size), seq_len * batch_size)
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
			rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), (rel_size, seq_len, seq_len, batch_size), 'F'))
			# batch_size x #dep x #head x #nclasses
	
		if isTrain or arc_targets is not None:
			loss = arc_loss + rel_loss
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
			outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len], arc_prob[np.arange(1, sent_len), arc_pred[1:sent_len]]))
		
		return outputs

	def save(self, save_path):
		self._pc.save(save_path)
	def load(self, load_path):
		self._pc.populate(load_path)
