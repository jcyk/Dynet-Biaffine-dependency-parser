#include <vector>
#include <unordered_map>

using namespace std;
class Graph
{
public:
	vector<vector<int> > out_edge;
	vector<vector<int> > in_edge;
	vector<vector<int> > JJ;
	vector<vector<float> > qq;
	unordered_map<long long, float> weights;
	void preprocess();
	vector<int> walk(int start_node, int walk_length);
	Graph();
	Graph(vector<int> u, vector<int> v, vector<float> w);
	~Graph();
};