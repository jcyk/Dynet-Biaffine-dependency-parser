from libcpp.vector cimport vector

cdef extern from "graph_c.h":
	cdef cppclass Graph:
		Graph () except +
		Graph (vector[int], vector[int], vector[float]) except +
		void preprocess()
		vector[int] walk(int, int)

cdef class PyGraph:
	cdef Graph c_graph
	def __cinit__(self, vector[int] u, vector[int] v, vector[float] w):
		self.c_graph = Graph(u, v, w)
	def preprocess(self):
		self.c_graph.preprocess()
	def walk(self, int start_node, int walk_length):
		return self.c_graph.walk(start_node, walk_length)