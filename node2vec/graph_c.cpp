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
		weights[key] = w[i];
	}
	out_edge.resize(mx+1);
	in_edge.resize(mx+1);
	for(int i =0;i<u.size();i++){
		out_edge[u[i]].push_back(v[i]);
		in_edge[v[i]].push_back(u[i]);
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
		J.resize(out_edge[u].size());
		q.resize(out_edge[u].size());
		for(int i=0; i<out_edge[u].size();i++){
			J[i] = 0;
			q[i] = 0;
		}
		for(int v : out_edge[u]){
			probs.push_back(weights[(long long)u<<31|v]);
		}
		norm(probs);

		/*vector<float> in_probs;
		int tmp = 0;
		for(int i = 0;i < out_edge[u].size();i++){
			int v = out_edge[u][i];
			for(int j =0; j< in_edge[v].size();j++){
				int w = in_edge[v][j];
				if (w == u) tmp = j;
				in_probs.push_back(weights[(long long)w<<31|v]);
			}
			norm(in_probs);
			probs[i] /= pow(in_probs[tmp], 2);
		}
		norm(probs);*/

		int K = probs.size();
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
		if (rand01()<qq[cur][kk])
			cur = out_edge[cur][kk];
		else
			cur = out_edge[cur][JJ[cur][kk]];
		res.push_back(cur);
	}
	return res;
}