#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include "Graph.hpp"

namespace clsa {

// 原有结构体定义保持不变
struct HRect {
    int x, y, w, h;
    bool operator==(const HRect &rhs) const {return x==rhs.x&&y==rhs.y&&w==rhs.w&&h==rhs.h;}
};

struct Rect {
    int w, h;
    bool operator==(const Rect &rhs) const {return w==rhs.w&&h==rhs.h;}
};

struct Conv2d {
    Rect stride, pad, krnlShape;
    bool operator==(const Conv2d &rhs) const {return stride==rhs.stride&&pad==rhs.pad&&krnlShape==rhs.krnlShape;}
    Rect ofmShape(Rect ifmShape) const {
        int h = 1 + (ifmShape.h - krnlShape.h + 2 * pad.h) / stride.h;
        int w = 1 + (ifmShape.w - krnlShape.w + 2 * pad.w) / stride.w;
        return {w, h};
    }
};

struct Layer {
    std::string layerName;
    Conv2d conv;
    bool operator==(const Layer &rhs) const {return layerName==rhs.layerName;}
};

struct LayerNode {
    Layer layer;
    Rect divSz;
    int x, y;
    bool operator==(const LayerNode &rhs) const {return layer==rhs.layer&&divSz==rhs.divSz&&x==rhs.x&&y==rhs.y;}
};

std::ostream &operator<<(std::ostream &ofs, const LayerNode &node) {
    ofs<<"( "<<node.layer.layerName<<": ("<<node.x<<", "<<node.y<<") )";
    return ofs;
}

struct LayerNodeHash {
    size_t operator()(const LayerNode &node) const {
        size_t seed = 0;
        std::hash<std::string> string_hasher;
        seed ^= string_hasher(node.layer.layerName) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.stride.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.stride.h) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.pad.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.pad.h) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.krnlShape.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.layer.conv.krnlShape.h) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(node.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

using LayerConf = std::vector<Layer>;

HRect OFMRect2IFM(HRect oRect, Conv2d conv) {
    int x1=oRect.x, y1=oRect.y, x2=oRect.x+oRect.w-1, y2=oRect.y+oRect.h-1;
    HRect ans;
    ans.x = x1*conv.stride.w - (conv.krnlShape.w-1)/2;
    ans.y = y1*conv.stride.h - (conv.krnlShape.h-1)/2;
    ans.w = (x2-x1)*conv.stride.w+conv.krnlShape.w;
    ans.h = (y2-y1)*conv.stride.h+conv.krnlShape.h;
    return ans;
}

// 配置加载类

struct LayerNodeOrd {
    using OrdT = std::unordered_map<LayerNode, int, LayerNodeHash>;
    
    LayerNodeOrd(const LayerConf &conf, Rect ifmShape, Rect divSz, int wdup=1, bool sdk=0, bool clsa=1)
        : m_conf{conf}, m_ifmShape{ifmShape}, m_divSz{divSz}, m_wdup{wdup}, m_SDK{sdk}, m_CLSA{clsa} {}
    
    OrdT get() {
        std::cout << "WDup: " << m_wdup << "\n";
        if (m_g.empty()) buildGraph();
        return m_g.topoSort();
    }
    
    int getCyclePerDiv() {
        return (m_divSz.w * m_divSz.h+m_wdup-1) / m_wdup;
    }
    
private:
    const LayerConf &m_conf;
    Graph<LayerNode, LayerNodeHash> m_g;
    Rect m_ifmShape;
    Rect m_divSz;
    int m_wdup;
    int m_maxOrd;
    bool m_SDK = 0;
    bool m_CLSA = 1;
    
    void buildGraph() {
        Rect curShape = m_ifmShape;
        for (int idx=0; idx<m_conf.size(); ++idx) {
            const auto &ofmConf = m_conf[idx];
            Rect ofmShape = ofmConf.conv.ofmShape(curShape);
            int ifmBoundX = (curShape.w-1)/m_divSz.w, ifmBoundY = (curShape.h-1)/m_divSz.h;
            int wdupBlk = (ofmShape.h/m_divSz.h+1+m_wdup-1)/m_wdup;
            if (!m_SDK) std::cout<<"BlkSz: "<<wdupBlk<<"\n";
            for (int i=0; i*m_divSz.h<ofmShape.h; ++i) {
                for (int j=0; j*m_divSz.w<ofmShape.w; ++j) {
                    HRect ofmRect = {j*m_divSz.w, i*m_divSz.h, m_divSz.w, m_divSz.h};
                    if (i>0&&j==0&&(!m_CLSA||m_SDK||i%wdupBlk)) {
                        int jM = (ofmShape.w/m_divSz.w);
                        m_g.add({m_conf[idx], m_divSz, jM, i-1}, {m_conf[idx], m_divSz, 0, i});
                    };
                    if (j>0) {
                        m_g.add({m_conf[idx], m_divSz, j-1, i}, {m_conf[idx], m_divSz, j, i});
                    }
                    if (m_CLSA) {
                        HRect ifmRect = OFMRect2IFM(ofmRect, ofmConf.conv);
                        LayerNode ofmNode = {m_conf[idx], m_divSz, j, i};
                        int x0 = std::max(0, ifmRect.x/m_divSz.w), x1 = std::min(ifmBoundX, (ifmRect.x+ifmRect.w-1)/m_divSz.w);
                        int y0 = std::max(0, ifmRect.y/m_divSz.h), y1 = std::min(ifmBoundY, (ifmRect.y+ifmRect.h-1)/m_divSz.h);
                        for (int x=x0; x<=x1; ++x)
                            for (int y=y0; y<=y1; ++y) {
                                LayerNode ifmNode = {idx>0?m_conf[idx-1]:Layer{"input"}, m_divSz, x, y};
                                m_g.add(ifmNode, ofmNode);
                            }
                    }
                }
            }
            if (!m_CLSA) {
                LayerNode ifmNode = {idx>0?m_conf[idx-1]:Layer{"input"}, m_divSz, ifmBoundX, ifmBoundY};
                LayerNode ofmNode = {m_conf[idx], m_divSz, 0, 0};
                m_g.add(ifmNode, ofmNode);
            }
            std::cout<<"curShape: ("<<curShape.w<<", "<<curShape.h<<")\n";
            curShape = ofmShape;
        }
    }
};

} // namespace clsa
