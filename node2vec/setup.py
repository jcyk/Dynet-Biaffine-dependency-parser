from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
	Extension('graph', ['graph.pyx', 'graph_c.cpp'],
		language='c++',
		extra_compile_args=["-std=c++11"],
		extra_link_args=["-std=c++11"]),
]
setup(
	ext_modules = cythonize(extensions)
)