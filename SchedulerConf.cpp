#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "Graph.hpp"
#include "CoreFunc.hpp"

namespace clsa {
// 配置加载类
class ConfigLoader {
public:
    static void loadGlobalConfig(const std::string& filename, Rect& inputShape, Rect& divSize, 
                                 int& wdup, bool& sdk, bool& clsa) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开配置文件: " + filename);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            if (line.find("input_shape:") == 0) {
                std::stringstream ss(line.substr(12));
                ss >> inputShape.w >> inputShape.h;
            } else if (line.find("div_size:") == 0) {
                std::stringstream ss(line.substr(9));
                ss >> divSize.w >> divSize.h;
            } else if (line.find("wdup:") == 0) {
                wdup = std::stoi(line.substr(5));
            } else if (line.find("sdk:") == 0) {
                sdk = std::stoi(line.substr(4));
            } else if (line.find("clsa:") == 0) {
                clsa = std::stoi(line.substr(5));
            }
        }
    }
    
    static LayerConf loadLayerConfig(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开层配置文件: " + filename);
        }
        
        LayerConf conf;
        std::string line;
        bool inLayersSection = false;
        
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            if (line == "layers:") {
                inLayersSection = true;
                continue;
            }
            
            if (inLayersSection) {
                std::stringstream ss(line);
                std::string name;
                int stride_w, stride_h, pad_w, pad_h, kernel_w, kernel_h;
                
                ss >> name >> stride_w >> stride_h >> pad_w >> pad_h >> kernel_w >> kernel_h;
                if (!name.empty() && ss.eof()) {
                    conf.push_back({
                        name,
                        {
                            {stride_w, stride_h},
                            {pad_w, pad_h},
                            {kernel_w, kernel_h}
                        }
                    });
                }
            }
        }
        
        if (conf.empty()) {
            throw std::runtime_error("未加载到任何网络层配置");
        }
        
        return conf;
    }
    
private:
    static std::string trim(const std::string& str) {
        size_t start = 0;
        while (start < str.size() && std::isspace(str[start])) start++;
        
        size_t end = str.size();
        while (end > start && std::isspace(str[end - 1])) end--;
        
        return str.substr(start, end - start);
    }
};

} // namespace clsa

int main(int argc, char* argv[]) {
    using namespace clsa;
    
    // 检查命令行参数
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <配置文件名> <输出文件名>" << std::endl;
        std::cerr << "示例: " << argv[0] << " network.conf output.txt" << std::endl;
        return 1;
    }
    
    const std::string configFile = argv[1];
    const std::string outputFile = argv[2];
    
    // 打开输出文件
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "错误: 无法打开输出文件 '" << outputFile << "'" << std::endl;
        return 1;
    }
    
    // 重定向 cout 到文件，保留 cerr 输出到控制台
    std::streambuf* originalCoutBuffer = std::cout.rdbuf(outFile.rdbuf());
    
    try {
        // 从配置文件读入全局参数
        Rect inputShape, divSize;
        int wdup;
        bool sdk, clsa;
        
        ConfigLoader::loadGlobalConfig(configFile, inputShape, divSize, wdup, sdk, clsa);
        
        // 从配置文件读入网络层配置
        LayerConf conf = ConfigLoader::loadLayerConfig(configFile);
        
        // 创建调度计算器
        auto ordCalc = LayerNodeOrd(conf, inputShape, divSize, wdup, sdk, clsa);
        auto ord = ordCalc.get();
        
        std::vector<std::pair<int, LayerNode>> ordVec;
        for (const auto [k, v] : ord)
            ordVec.push_back({v, k});
            
        std::sort(ordVec.begin(), ordVec.end(), 
                  [](const auto &lhs, const auto &rhs){ return lhs.first < rhs.first; });
        
        if (!ordVec.empty()) {
            int cyclePerDiv = ordCalc.getCyclePerDiv();
            std::cout << "Cycle per div: " << cyclePerDiv << "\n";
            std::cout << "Total cycles: " << ordVec.back().first * cyclePerDiv << "\n";
        }
        
        std::cout << "Schedule Table (Node, Ord): \n";
        for (const auto [k, v] : ordVec)
            if (v.layer.layerName != "input")
                std::cout << v << ": " << k << "\n";
                
    } catch (const std::exception& e) {
        // 恢复 cout 缓冲区
        std::cout.rdbuf(originalCoutBuffer);
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 恢复 cout 缓冲区
    std::cout.rdbuf(originalCoutBuffer);
    std::cout << "调度表已生成并保存到 '" << outputFile << "'" << std::endl;
    
    return 0;
}
