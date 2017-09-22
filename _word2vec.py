import gensim
import sys

def write_sentences(of, fname):
	sent = []
	for line in open(fname).readlines():
		info = line.strip().split()
		if info:
			assert(len(info)==10), 'Illegal line: %s'%line
			word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
			sent.append([word, tag, head, rel])
		else:
			of.write(' '.join([w[0] for w in sent])+'\n')
			sent = []

def create_graph(file_list):
	of = open('corpus', 'w')
	for fname in file_list:
		write_sentences(of, fname)
	of.close()


class MySentences(object):
	def __init__(self,filename):
		self.filename = filename
	def __iter__(self):
		for line in open(self.filename):
			yield line.strip().split()

if __name__ == "__main__":
	create_graph(sys.argv[1:])
	sents = MySentences('corpus')
	model = gensim.models.Word2Vec(sents, size=100, window=5, min_count=2, workers = 4, iter=10)
	model.wv.save_word2vec_format('word2vec_emb')

