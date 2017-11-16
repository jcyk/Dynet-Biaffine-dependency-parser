#java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile rawTextToParse -outFile dependenciesOutputFile.txi -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np

from lib import biLSTM, leaky_relu, bilinear, orthonormal_initializer, arc_argmax, rel_argmax, orthonormal_VanillaLSTMBuilder, softplus

class VAEParser(object):
	def __init__(self, vocab,
					   word_dims,
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
		
		self.LSTM_builders = []
		f = orthonormal_VanillaLSTMBuilder(1, word_dims, lstm_hiddens, pc, randn_init)
		b = orthonormal_VanillaLSTMBuilder(1, word_dims, lstm_hiddens, pc, randn_init)
		self.LSTM_builders.append((f,b))
		for i in xrange(lstm_layers-1):
			f = orthonormal_VanillaLSTMBuilder(1, 2*lstm_hiddens, lstm_hiddens, pc, randn_init)
			b = orthonormal_VanillaLSTMBuilder(1, 2*lstm_hiddens, lstm_hiddens, pc, randn_init)
			self.LSTM_builders.append((f,b))
		self.dropout_lstm_input = dropout_lstm_input
		self.dropout_lstm_hidden = dropout_lstm_hidden

		mlp_size = mlp_arc_size+mlp_rel_size
		self.mlp_size = mlp_size
		#self.hW = pc.parameters_from_numpy(orthonormal_initializer(2*lstm_hiddens, 2*lstm_hiddens, randn_init))
		#self.hb = pc.add_parameters((2*lstm_hiddens,), init = dy.ConstInitializer(0.))
		self.rhW = pc.parameters_from_numpy(orthonormal_initializer(mlp_size, mlp_size, randn_init))
		self.rhb = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.rdW = pc.parameters_from_numpy(orthonormal_initializer(mlp_size, mlp_size, randn_init))
                self.rdb = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		W = orthonormal_initializer(mlp_size, 2*lstm_hiddens, randn_init)
		self.mlp_dep_Wm = pc.parameters_from_numpy(W)
		self.mlp_head_Wm = pc.parameters_from_numpy(W)
		self.mlp_dep_bm = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_head_bm = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))

		W = orthonormal_initializer(mlp_size, 2*lstm_hiddens, randn_init)
		self.mlp_dep_Wv = pc.parameters_from_numpy(W)
		self.mlp_head_Wv = pc.parameters_from_numpy(W)
		self.mlp_dep_bv = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_head_bv = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_arc_size = mlp_arc_size
		self.mlp_rel_size = mlp_rel_size
		self.dropout_mlp = dropout_mlp

		self.arc_W = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init = dy.ConstInitializer(0.))
		self.rel_W = pc.add_parameters((vocab.rel_size*(mlp_rel_size +1) , mlp_rel_size + 1), init = dy.ConstInitializer(0.))

		self._pc = pc
		self.dropout_emb = dropout_emb
		self.fixed_word_emb = False
		self.use_pret = True

	@property 
	def parameter_collection(self):
		return self._pc

	def run(self, word_inputs, arc_targets = None, rel_targets = None, isTrain = True):
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
		
		if not self.use_pret:
			word_embs = [dy.lookup_batch(self.word_embs, np.where( w<self._vocab.words_in_train, w, self._vocab.UNK)) for w in word_inputs ]
		else:
			if self.fixed_word_emb:
				word_embs = [dy.lookup_batch(self.pret_word_embs, w, update = False) for w in word_inputs ]
			else:
				word_embs = [dy.lookup_batch(self.word_embs, np.where( w<self._vocab.words_in_train, w, self._vocab.UNK)) + dy.lookup_batch(self.pret_word_embs, w, update = False) for w in word_inputs]
		
		if isTrain:
			word_embs= [ dy.dropout_dim(w, 0, self.dropout_emb) for w in word_embs]

		top_recur = dy.concatenate_cols(biLSTM(self.LSTM_builders, word_embs, batch_size, self.dropout_lstm_input if isTrain else 0., self.dropout_lstm_hidden if isTrain else 0.))
		if isTrain:
			top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

		#Wh = dy.parameter(self.hW)
		#bh = dy.parameter(self.hb)
		#top_recur = dy.logistic(dy.affine_transform([bh, Wh, top_recur]))
		Wm_dep, bm_dep = dy.parameter(self.mlp_dep_Wm), dy.parameter(self.mlp_dep_bm)
		Wm_head, bm_head = dy.parameter(self.mlp_head_Wm), dy.parameter(self.mlp_head_bm)
		dep_m, head_m = dy.affine_transform([bm_dep, Wm_dep, top_recur]), dy.affine_transform([bm_head, Wm_head, top_recur])
		
		if isTrain:
			Wv_dep, bv_dep = dy.parameter(self.mlp_dep_Wv), dy.parameter(self.mlp_dep_bv)
			Wv_head, bv_head = dy.parameter(self.mlp_head_Wv), dy.parameter(self.mlp_head_bv)
			dep_v, head_v = softplus(dy.affine_transform([bv_dep, Wv_dep, top_recur])),  softplus(dy.affine_transform([bv_head, Wv_head, top_recur]))
	
			#print dep_v.npvalue(), head_v.npvalue()	
			eps = dy.random_normal((self.mlp_size, seq_len), batch_size)
			dep = dep_m + dy.cmult(dy.sqrt(dep_v), eps)

			eps = dy.random_normal((self.mlp_size, seq_len), batch_size)
			head = head_m + dy.cmult(dy.sqrt(head_v), eps)

			KL_dep = -0.5*dy.sum_dim(1 + dy.log(dep_v + 1e-6) - dy.square(dep_m) - dep_v, [0])
			KL_head = -0.5*dy.sum_dim(1 + dy.log(head_v + 1e-6) - dy.square(head_m) - head_v, [0])
			KL_loss = dy.sum_batches( dy.reshape(KL_dep + KL_head, (1,), seq_len*batch_size) * mask_1D_tensor) / num_tokens
			#print KL_loss.scalar_value(),' '
		else:
			dep = dep_m
			head = head_m

		#if isTrain:
		#	dep, head= dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)
		Wrd, brd =dy.parameter(self.rdW),dy.parameter(self.rdb)
		Wrh, brh = dy.parameter(self.rhW), dy.parameter(self.rhb)
		
		dep = leaky_relu(dy.affine_transform([brd, Wrd, dep]))
		head = leaky_relu(dy.affine_transform([brh, Wrh, head]))		
		dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
		head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

		W_arc = dy.parameter(self.arc_W)
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
			loss = arc_loss + rel_loss #+ KL_loss
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

	def save(self, save_path):
		self._pc.save(save_path)
	def load(self, load_path):
		self._pc.populate(load_path)
