'''
Author: Deng Cai

Adapted from

Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from word2vec import Word2Vec
from collections import Counter
import sys, random

ROOT = '<root>'
depth_cnt = Counter()
length_cnt = Counter()

def update_graph(graph, fname):
	nsents = 0
	sent = [[ROOT, ROOT, 0, ROOT]]
	for line in open(fname).readlines():
		info = line.strip().split()
		if info:
			assert(len(info)==10), 'Illegal line: %s'%line
			word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
			sent.append([word, tag, head, rel])
		else:
			for idx, (word, tag, head, rel) in enumerate(sent[1:],1):
				depth = 1
				h = head
				while h!=0:
					h = sent[h][2]
					depth +=1
				depth_cnt[depth] +=1
				length_cnt[abs(idx-head)] +=1
			nsents += 1
			graph.update([(sent[head][0], word) for word, tag, head, rel in sent[1:]])
			sent = [[ROOT, ROOT, 0, ROOT]]
	return nsents

def create_graph(file_list):
	graph = Counter()
	nsents = 0
	for fname in file_list:
		nsents += update_graph(graph, fname)
	print 'depths of dependencies', depth_cnt
	print 'lengths of dependencies', length_cnt
	print 'number of sentences', nsents
	return nsents, graph

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='+', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=100,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=20,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	nsents, graph = create_graph(args.input)
	#for edge in graph:
	#	print edge[0], edge[1], graph[edge]
	
	G = nx.DiGraph()

	if args.weighted:
		for edge in graph:
			if graph[edge] >2:
				G.add_edge(edge[0], edge[1],{'weight':graph[edge]})
	else:
		for edge in graph:
			if graph[edge] >2:
				G.add_edge(edge[0], edge[1])
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return nsents, G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)

class Simulate_walks(object):
	def __init__(self, G, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		self.G = G
		self.num_walks = num_walks
		self.walk_length = walk_length
	
	def __iter__(self):
		#nodes = list(self.G.G.nodes())
		node = '<root>'
		for walk_iter in range(self.num_walks):
			yield self.G.node2vec_walk(walk_length=self.walk_length, start_node=node)
			#random.shuffle(nodes)
			#for node in nodes:
			#	yield self.G.node2vec_walk(walk_length=self.walk_length, start_node=node)

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nsents, nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	print G.G.number_of_nodes(), G.G.number_of_edges()
	G.preprocess_transition_probs()

	print 'Graph preprocessed'
	walks = Simulate_walks(G, args.num_walks * nsents, args.walk_length)
	learn_embeddings(walks)


if __name__ == "__main__":
	args = parse_args()
	main(args)
