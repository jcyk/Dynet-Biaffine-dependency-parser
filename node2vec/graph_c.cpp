#include "graph_c.h"
#include <algorithm>
#include <unordered_set>
#include <random>
#include <iostream>
#include <math.h>
using namespace std;

Graph::Graph() {}
Graph::Graph(vector<int> u, vector<int> v, vector<float> w)
{
	int mx = 0;
	for(int i =0;i <u.size();i++){
		mx = max(mx, max(u[i], v[i]));
		long long key = (long long)u[i]<<31|v[i];
		if (weights.find(key) != weights.end()) weights[key] += w[i];
		else weights[key] = w[i];
	}
	nodes = mx +1;
	edges = weights.size();
	cout << "number of nodes " << nodes << endl;
	cout << "number of edges "<< edges << endl;
	out_edge.resize(mx+1);
	in_edge.resize(mx+1);
	unordered_set<long long> tmp;
	for(int i =0;i<u.size();i++){
		long long key = (long long)u[i]<<31|v[i];
		if (tmp.find(key) == tmp.end()){
			out_edge[u[i]].push_back(v[i]);
			in_edge[v[i]].push_back(u[i]);
			tmp.insert(key);
		}
	}
}

Graph::~Graph() {}

void norm(vector<float>& vec)
{
	float tot = 0.;
	for(float x : vec)
		tot += x;
	for(float& x: vec)
		x/=tot;
}
void Graph::preprocess()
{
	vector<float> probs;
	vector<int> smaller;
	vector<int> larger;
	vector<int> J;
	vector<float> q; 
	for(int u =0;u < out_edge.size();u++){
		probs.clear();
		smaller.clear();
		larger.clear();
		int K = out_edge[u].size();

		J.resize(K);
		q.resize(K);
		for(int i=0; i< K;i++){
			J[i] = 0;
			q[i] = 0;
		}
		for(int v : out_edge[u]){
			probs.push_back(weights[(long long)u<<31|v]*log(nodes/in_edge[v].size()));
		}
		norm(probs);
		for(int i =0;i<K;i++){
			q[i] = K*probs[i];
			if (q[i]<1.)
				smaller.push_back(i);
			else
				larger.push_back(i);
		}
		while (smaller.size()>0 && larger.size()>0){
			int small = smaller.back();
			smaller.pop_back();
			int large = larger.back();
			larger.pop_back();
			J[small] = large;
			q[large] = q[large] + q[small] - 1.;
			if (q[large]<1.)
				smaller.push_back(large);
			else
				larger.push_back(large);
		}
		JJ.push_back(J);
		qq.push_back(q);
	}
}

float rand01()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}

vector<int> Graph::walk(int start_node, int walk_length){
	vector<int> res;
	res.push_back(start_node);
	int cur = start_node;
	while (res.size()<walk_length){
		if (out_edge[cur].size() == 0) break;
		int kk = rand01()*JJ[cur].size();
		if (kk == JJ[cur].size()) kk--;
		if (rand01()<qq[cur][kk])
			cur = out_edge[cur][kk];
		else
			cur = out_edge[cur][JJ[cur][kk]];
		res.push_back(cur);
	}
	return res;
}