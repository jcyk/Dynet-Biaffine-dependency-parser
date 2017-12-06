# -*- coding: UTF-8 -*-
import dynet as dy
import numpy as np
from data import Vocab
from tarjan import Tarjan

def softplus(x):
	return dy.log(dy.exp(x) + 1.)

def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc, randn_init = False):
	builder = dy.CompactVanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
	for layer, params in enumerate(builder.get_parameters()):
		W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer >0 else input_dims), randn_init)
		W_h, W_x = W[:,:lstm_hiddens], W[:,lstm_hiddens:]
		params[0].set_value(np.concatenate([W_x]*4, 0))
		params[1].set_value(np.concatenate([W_h]*4, 0))
		b = np.zeros(4*lstm_hiddens, dtype=np.float32)
		params[2].set_value(b)
	return builder
	
def uniLSTM(builder, inputs, batch_size = None, dropout_x = 0., dropout_h = 0., update = True):
	h0 = builder.initial_state(update = update)
	builder.set_dropouts(dropout_x, dropout_h)
	if batch_size is not None:
		builder.set_dropout_masks(batch_size)
	hs = h0.transduce(inputs)
	return hs

def biLSTM(builders, inputs, batch_size = None, dropout_x = 0., dropout_h = 0., update = True):
	for fb, bb in builders:
		f, b = fb.initial_state(update = update), bb.initial_state(update = update)
		fb.set_dropouts(dropout_x, dropout_h)
		bb.set_dropouts(dropout_x, dropout_h)
		if batch_size is not None:
			fb.set_dropout_masks(batch_size)
			bb.set_dropout_masks(batch_size)
		fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
		inputs = [dy.concatenate([f,b]) for f, b in zip(fs, reversed(bs))]
	return inputs

def leaky_relu(x):
	return dy.bmax(.1*x, x)

def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs = 1, bias_x = False, bias_y = False):
	# x,y: (input_size x seq_len) x batch_size
	if bias_x:
		x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
	if bias_y:
		y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
	
	nx, ny = input_size + bias_x, input_size + bias_y
	# W: (num_outputs x ny) x nx
	lin = W * x
	if num_outputs > 1:
		lin = dy.reshape(lin, (ny, num_outputs*seq_len), batch_size = batch_size)
	blin = dy.transpose(y) * lin
	if num_outputs > 1:
		blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size = batch_size)
	# seq_len_y x seq_len_x if output_size == 1
	# seq_len_y x num_outputs x seq_len_x else
	return blin


def orthonormal_initializer(output_size, input_size, randn_init = False):
	"""
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
	"""
	if randn_init:
		return np.random.randn(output_size, input_size).astype(np.float32)
	I = np.eye(output_size)
	lr = .1
	eps = .05/(output_size + input_size)
	success = False
	tries = 0
	while not success and tries < 10:
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
		for i in xrange(100):
			QTQmI = Q.T.dot(Q) - I
			loss = np.sum(QTQmI**2 / 2)
			Q2 = Q**2
			Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
			if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
				tries += 1
				lr /= 2
				break
		success = True
	print (output_size, input_size)
	if success:
		print('Orthogonal pretrainer loss: %.2e' % loss)
	else:
		print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
	return np.transpose(Q.astype(np.float32))

def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree = True):
	"""
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
	"""
	if ensure_tree:
		I = np.eye(len(tokens_to_keep))
		# block loops and pad heads
		parse_probs = parse_probs * tokens_to_keep * (1-I)
		parse_preds = np.argmax(parse_probs, axis=1)
		tokens = np.arange(1, length)
		roots = np.where(parse_preds[tokens] == 0)[0]+1
		# ensure at least one root
		if len(roots) < 1:
			# The current root probabilities
			root_probs = parse_probs[tokens,0]
			# The current head probabilities
			old_head_probs = parse_probs[tokens, parse_preds[tokens]]
			# Get new potential root probabilities
			new_root_probs = root_probs / old_head_probs
			# Select the most probable root
			new_root = tokens[np.argmax(new_root_probs)]
			# Make the change
			parse_preds[new_root] = 0
			# ensure at most one root
		elif len(roots) > 1:
			# The probabilities of the current heads
			root_probs = parse_probs[roots,0]
			# Set the probability of depending on the root zero
			parse_probs[roots,0] = 0
			# Get new potential heads and their probabilities
			new_heads = np.argmax(parse_probs[roots][:,tokens], axis=1)+1
			new_head_probs = parse_probs[roots, new_heads] / root_probs
			# Select the most probable root
			new_root = roots[np.argmin(new_head_probs)]
			# Make the change
			parse_preds[roots] = new_heads
			parse_preds[new_root] = 0
		# remove cycles
		tarjan = Tarjan(parse_preds, tokens)
		cycles = tarjan.SCCs
		for SCC in tarjan.SCCs:
			if len(SCC) > 1:
				dependents = set()
				to_visit = set(SCC)
				while len(to_visit) > 0:
					node = to_visit.pop()
					if not node in dependents:
						dependents.add(node)
						to_visit.update(tarjan.edges[node])
				# The indices of the nodes that participate in the cycle
				cycle = np.array(list(SCC))
				# The probabilities of the current heads
				old_heads = parse_preds[cycle]
				old_head_probs = parse_probs[cycle, old_heads]
				# Set the probability of depending on a non-head to zero
				non_heads = np.array(list(dependents))
				parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
				# Get new potential heads and their probabilities
				new_heads = np.argmax(parse_probs[cycle][:,tokens], axis=1)+1
				new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
				# Select the most probable change
				change = np.argmax(new_head_probs)
				changed_cycle = cycle[change]
				old_head = old_heads[change]
				new_head = new_heads[change]
				# Make the change
				parse_preds[changed_cycle] = new_head
				tarjan.edges[new_head].add(changed_cycle)
				tarjan.edges[old_head].remove(changed_cycle)
		return parse_preds
	else:
		# block and pad heads
		parse_probs = parse_probs * tokens_to_keep
		parse_preds = np.argmax(parse_probs, axis=1)
		return parse_preds

def rel_argmax(rel_probs, length, ensure_tree = True):
	"""
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
	"""
	if ensure_tree:
		rel_probs[:,Vocab.PAD] = 0
		root = Vocab.ROOT
		tokens = np.arange(1, length)
		rel_preds = np.argmax(rel_probs, axis=1)
		roots = np.where(rel_preds[tokens] == root)[0]+1
		if len(roots) < 1:
			rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
		elif len(roots) > 1:
			root_probs = rel_probs[roots, root]
			rel_probs[roots, root] = 0
			new_rel_preds = np.argmax(rel_probs[roots], axis=1)
			new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
			new_root = roots[np.argmin(new_rel_probs)]
			rel_preds[roots] = new_rel_preds
			rel_preds[new_root] = root
		return rel_preds
	else:
		rel_probs[:,Vocab.PAD] = 0
		rel_preds = np.argmax(rel_probs, axis=1)
		return rel_preds