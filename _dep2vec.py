from collections import Counter
import sys, random

ROOT = '<root>'
def update_graph(graph, fname):
	sent = [[ROOT, ROOT, 0, ROOT]]
	for line in open(fname).readlines():
		info = line.strip().split()
		if info:
			assert(len(info)==10), 'Illegal line: %s'%line
			word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
			sent.append([word, tag, head, rel])
		else:
			for word, tag, head, rel in sent[1:]:
				if rel == 'prep':
					continue
				if rel == 'pobj' and head !=0:
					h = sent[sent[head][2]][0]
					rel = "%s:%s"%(sent[head][-1], sent[head][0])
				else:
					h = sent[head][0]
				graph.append((h, word, rel))
			sent = [[ROOT, ROOT, 0, ROOT]]

def create_graph(file_list, typed):
	graph = []
	for fname in file_list:
		update_graph(graph, fname)
	edge_cnt = Counter()
	edge_cnt.update(graph)

	edge_file = open('dep.contexts', 'w')
	for edge in graph:
		if edge_cnt[edge] < 3:
			continue
		if typed:
			out = "%s %s_%s\n%s %sI_%s\n"%(edge[0],edge[2],edge[1], edge[1], edge[2],edge[0])
		else:
			out = "%s _%s\n%s I_%s\n"%(edge[0],edge[1], edge[1],edge[0])
		edge_file.write(out)
	edge_file.close()

	data = [line.strip().split() for line in  open('dep.contexts').readlines()]
	wv_cnt = Counter()
	cv_cnt = Counter()

	wv = wv_cnt.update([x[0] for x in data])
	cv = cv_cnt.update((x[1] for x in data))


	with open('wv', 'w') as wv_file:
		wv_file.write('</s> 0\n')
		for x in wv_cnt.most_common():
			if x[1] >= 5:
				wv_file.write('%s %d\n'%(x[0],x[1]))

	with open('cv', 'w') as cv_file:
		cv_file.write('</s> 0\n')
		for x in cv_cnt.most_common():
			if x[1] >= 5:
				cv_file.write('%s %d\n'%(x[0],x[1]))


if __name__ == "__main__":
	create_graph(sys.argv[1:], typed  = False)
