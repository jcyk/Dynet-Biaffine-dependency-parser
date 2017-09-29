# -*- coding: UTF-8 -*-
from __future__ import division
from collections import Counter
import numpy as np

from k_means import KMeans

class Vocab(object):
	PAD, ROOT, UNK = 0, 1, 2

	def __init__(self, input_file, pret_file = None, min_occur_count = 2, aux_pret_file = None):
		word_counter = Counter()
		tag_set = set()
		rel_set = set()
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==10), 'Illegal line: %s'%line
					word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
					word_counter[word] += 1
					tag_set.add(tag)
					if rel !='root':
						rel_set.add(rel)

		self._id2word = ['<pad>', '<root>', '<unk>']
		self._id2tag = ['<pad>', '<root>', '<unk>']
		self._id2rel = ['<pad>', 'root']
		for word, count in word_counter.most_common():
			if count > min_occur_count:
				self._id2word.append(word)
		self._words_in_train_data = len(self._id2word)
		self._pret_file = pret_file
		if pret_file:
			self._add_pret_words(pret_file)
		self._words_in_train_or_pret_data = len(self._id2word)

		self._aux_pret_file = aux_pret_file
		if aux_pret_file:
			self._add_aux_pret_words(aux_pret_file)

		self._id2tag += list(tag_set)
		self._id2rel += list(rel_set)

		reverse = lambda x : dict(zip(x, range(len(x))))
		self._word2id = reverse(self._id2word)
		self._tag2id = reverse(self._id2tag)
		self._rel2id = reverse(self._id2rel)
		print "Vocab info: #words %d, #tags %d #rels %d"%(self.vocab_size,self.tag_size, self.rel_size)
	
	def merge_with(self, o):
		self._id2word = ['<pad>', '<root>', '<unk>'] + list(set(self._id2word[3:self.words_in_train]+ o._id2word[3:o.words_in_train]))
		self._id2tag = ['<pad>', '<root>', '<unk>'] + list(set(self._id2tag[3:]+self._id2tag[3:]))
		self._id2rel = ['<pad>', 'root'] + list(set(self._id2rel[2:] + self._id2rel[2:]))
		self._words_in_train_data = len(self._id2word)
		if self._pret_file:
			self._add_pret_words(self._pret_file)
		if self._aux_pret_file:
			self._add_aux_pret_words(self._aux_pret_file)

		reverse = lambda x : dict(zip(x, range(len(x))))
		self._word2id = reverse(self._id2word)
		self._tag2id = reverse(self._id2tag)
		self._rel2id = reverse(self._id2rel)	

	def _add_pret_words(self, pret_file):
		print '#words in training set:', self._words_in_train_data
		words_in_train_data = set(self._id2word)
		with open(pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word = line[0]
					if word not in words_in_train_data:
						self._id2word.append(word)
		#print 'Total words:', len(self._id2word)

	def get_pret_embs(self):
		assert (self._pret_file is not None), "No pretrained file provided."
		embs = [[]] * self._words_in_train_or_pret_data
		with open(self._pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word, data = line[0], line[1:]
					embs[self._word2id[word]] = data
		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		pret_embs = np.array(embs, dtype=np.float32)
		return pret_embs / np.std(pret_embs)


	def get_word_embs(self, word_dims):
		if self._pret_file is not None:
			return np.zeros((self.words_in_train, word_dims), dtype=np.float32)
		return np.random.randn(self.words_in_train, word_dims).astype(np.float32)

	def get_tag_embs(self, tag_dims):
		return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

	def word2id(self, xs):
		if isinstance(xs, list):
			return [self._word2id.get(x, self.UNK) for x in xs]
		return self._word2id.get(xs, self.UNK)
	
	def id2word(self, xs):
		if isinstance(xs, list):
			return [self._id2word[x] for x in xs]
		return self._id2word[xs]

	def rel2id(self, xs):
		if isinstance(xs, list):
			return [self._rel2id.get(x, self.PAD) for x in xs]
		return self._rel2id.get(xs, self.PAD)

	def id2rel(self,xs):
		if isinstance(xs, list):
			return [self._id2rel[x] for x in xs]
		return self._id2rel[xs]

	def tag2id(self, xs):
		if isinstance(xs, list):
			return [self._tag2id.get(x, self.UNK) for x in xs]
		return self._tag2id.get(xs, self.UNK)

	@property 
	def words_in_train(self):
		return self._words_in_train_data
	
	@property 
	def words_in_train_or_pret(self):
		return self._words_in_train_or_pret_data

	@property
	def vocab_size(self):
		return len(self._id2word)

	@property
	def tag_size(self):
		return len(self._id2tag)

	@property
	def rel_size(self):
		return len(self._id2rel)


	def _add_aux_pret_words(self, input_file):
		print '#words known', self._words_in_train_or_pret_data
		known_words = set(self._id2word)
		start = True
		with open(input_file) as f:
			for line in f.readlines():
				if start:
					start = False
					continue
				line = line.strip().split()
				if line:
					word = line[0]
					if word not in known_words:
						self._id2word.append(word)

	def get_aux_pret_embs(self):
		assert (self._aux_pret_file is not None), "No auxiliary pretrained file provided."
		embs = [[]] * len(self._id2word) 
		start = True
		with open(self._aux_pret_file) as f:
			for line in f.readlines():
				if start:
					start = False
					continue
				line = line.strip().split()
				if line:
					word, data = line[0], line[1:]
					embs[self._word2id[word]] = data
		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		pret_embs = np.array(embs, dtype=np.float32)
		return pret_embs


class DataLoader(object):
	def __init__(self, input_file, n_bkts, vocab, bucket_sizes = None):
		sents = []
		sent = [[Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT]]
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==10), 'Illegal line: %s'%line
					word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(info[6]), vocab.rel2id(info[7])
					sent.append([word, tag, head, rel])
				else:
					sents.append(sent)
					sent = [[Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT]]
		
		len_counter = Counter()
		for sent in sents:
			len_counter[len(sent)] += 1	
		if bucket_sizes is None:
			self._bucket_sizes = KMeans(n_bkts, len_counter).splits
		else:
			self._bucket_sizes = bucket_sizes
		self._buckets = [[] for i in xrange(n_bkts)]
		len2bkt = {}
		prev_size = -1
		for bkt_idx, size in enumerate(self._bucket_sizes):
			len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
			prev_size = size

		self._record = []
		for sent in sents:
			bkt_idx = len2bkt[len(sent)]
			self._buckets[bkt_idx].append(sent)
			idx = len(self._buckets[bkt_idx]) - 1
			self._record.append((bkt_idx, idx))

		for bkt_idx, (bucket, size) in enumerate(zip(self._buckets, self._bucket_sizes)):
			self._buckets[bkt_idx] = np.zeros((size, len(bucket), 4), dtype=np.int32)
			for idx, sent in enumerate(bucket):
				self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)
	@property
	def idx_sequence(self):
		return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]
	
	def get_batches(self, batch_size, shuffle = True):
		batches = []
		for bkt_idx, bucket in enumerate(self._buckets):
			bucket_len = bucket.shape[1]
			n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
			n_splits = max(n_tokens // batch_size, 1)
			range_func = np.random.permutation if shuffle else np.arange
			for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
				batches.append((bkt_idx, bkt_batch))
		
		if shuffle:
			np.random.shuffle(batches)

		for bkt_idx, bkt_batch in batches:
			word_inputs = self._buckets[bkt_idx][:,bkt_batch, 0]
			tag_inputs = self._buckets[bkt_idx][:,bkt_batch, 1]
			arc_targets = self._buckets[bkt_idx][:,bkt_batch, 2]
			rel_targets = self._buckets[bkt_idx][:,bkt_batch, 3]
			yield word_inputs, tag_inputs, arc_targets, rel_targets

class MixedDataLoader(object):
	def __init__(self, files, ratios, n_bkts, vocab):
		assert isinstance(files, list)
		assert isinstance(ratios, list)
		test = DataLoader(files[0], n_bkts, vocab)
		bucket_sizes = test._bucket_sizes
		del test
		self._loaders = [ DataLoader(_file, n_bkts, vocab, bucket_sizes = bucket_sizes) for _file in files]
		self._ratios = ratios

	def get_batches(self, batch_size):
		generators = [loader.get_batches(int(ratio * batch_size)) for loader, ratio in zip(self._loaders, self._ratios)]
		for batch_tuple in zip(*generators):
			yield batch_tuple

class RawDataLoader(object):
	def __init__(self, input_file, n_bkts, vocab):
		sents = []
		sent = [[Vocab.ROOT, Vocab.ROOT]]
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==2), 'Illegal line: %s'%line
					word, tag = vocab.word2id(info[0].lower()), vocab.tag2id(info[1])
					sent.append([word, tag])
				else:
					sents.append(sent)
					sent = [[Vocab.ROOT, Vocab.ROOT]]		
		len_counter = Counter()
		for sent in sents:
			len_counter[len(sent)] += 1	
		self._bucket_sizes = KMeans(n_bkts, len_counter).splits
		self._buckets = [[] for i in xrange(n_bkts)]
		len2bkt = {}
		prev_size = -1
		for bkt_idx, size in enumerate(self._bucket_sizes):
			len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
			prev_size = size

		self._record = []
		for sent in sents:
			bkt_idx = len2bkt[len(sent)]
			self._buckets[bkt_idx].append(sent)
			idx = len(self._buckets[bkt_idx]) - 1
			self._record.append((bkt_idx, idx))

		for bkt_idx, (bucket, size) in enumerate(zip(self._buckets, self._bucket_sizes)):
			self._buckets[bkt_idx] = np.zeros((size, len(bucket), 2), dtype=np.int32)
			for idx, sent in enumerate(bucket):
				self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)
	@property
	def idx_sequence(self):
		return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]
	
	def get_batches(self, batch_size, shuffle = True):
		batches = []
		for bkt_idx, bucket in enumerate(self._buckets):
			bucket_len = bucket.shape[1]
			n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
			n_splits = max(n_tokens // batch_size, 1)
			range_func = np.random.permutation if shuffle else np.arange
			for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
				batches.append((bkt_idx, bkt_batch))
		
		if shuffle:
			np.random.shuffle(batches)

		finished_batches = 0
		for bkt_idx, bkt_batch in batches:
			print finished_batches,'/',len(batches)
			finished_batches += 1
			word_inputs = self._buckets[bkt_idx][:,bkt_batch, 0]
			tag_inputs = self._buckets[bkt_idx][:,bkt_batch, 1]
			yield word_inputs, tag_inputs
