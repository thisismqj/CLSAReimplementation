#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <utility>
#include <algorithm>
// 不带权有向图
namespace clsa {
template <typename _TNode, typename _TNodeHash>
struct Graph {
    using AdjList = std::vector<_TNode>;
    std::unordered_map<_TNode, AdjList, _TNodeHash> m_g;
    std::unordered_map<_TNode, int, _TNodeHash> m_ind;
    bool empty() const {return m_g.empty();}
    void add(_TNode x, _TNode y) {
        if (std::find(m_g[x].begin(), m_g[x].end(), y)==m_g[x].end()) {
            m_g[x].push_back(y);
            if (m_ind.find(x)==m_ind.end()) m_ind[x]=0;
            ++m_ind[y];
        }
    }
    std::unordered_map<_TNode, int, _TNodeHash> topoSort() {
        std::unordered_map<_TNode, int, _TNodeHash> ans;
        auto ind = m_ind;
        std::queue<std::pair<_TNode, int>> Q;
        for (auto [n, d]: m_ind)
            if (d==0) Q.push({n, 0});
        while (!Q.empty()) {
            auto [n, d]=Q.front();
            Q.pop();
            ans[n]=d;
            for (auto y: m_g[n]) {
                --ind[y];
                if (ind[y]==0) Q.push({y, 1+d});
            }
        }
        return ans;
    }
} ;
}
