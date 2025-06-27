#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>
#include <omp.h>
#include <chrono>
#include <iomanip>

const double INF = std::numeric_limits<double>::infinity();

// 函数：读取邻接表文件
bool read_adjacency_list(const std::string& filename, int& num_vertices, std::vector<std::tuple<int, int, double>>& edges) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open adjacency list file: " << filename << std::endl;
        return false;
    }

    std::string line;
    num_vertices = 0;
    // 先读取一遍以确定顶点数量
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int u, v;
        double weight;
        if (iss >> u >> v >> weight) {
            edges.emplace_back(u, v, weight);
            num_vertices = std::max({num_vertices, u + 1, v + 1}); // 假设顶点ID从0开始
        } else {
            std::cerr << "Warning: Skipping invalid line in adjacency list: " << line << std::endl;
        }
    }
    infile.close();
    return true;
}

// 函数：读取测试用例文件
bool read_test_cases(const std::string& filename, std::vector<std::pair<int, int>>& test_pairs) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open test case file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            test_pairs.emplace_back(u, v);
        } else {
            std::cerr << "Warning: Skipping invalid line in test case file: " << line << std::endl;
        }
    }
    infile.close();
    return true;
}


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <adjacency_list_file> <test_cases_file> <num_threads>" << std::endl;
        return 1;
    }

    std::string adj_list_filename = argv[1];
    std::string test_cases_filename = argv[2];
    int num_threads = std::stoi(argv[3]);

    if (num_threads <= 0 || num_threads > 16) {
        std::cerr << "Error: Number of threads must be between 1 and 16." << std::endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    int num_vertices;
    std::vector<std::tuple<int, int, double>> edges;

    if (!read_adjacency_list(adj_list_filename, num_vertices, edges)) {
        return 1;
    }

    if (num_vertices == 0) {
        std::cout << "No vertices found in the graph." << std::endl;
        return 0;
    }

    std::vector<std::vector<double>> dist(num_vertices, std::vector<double>(num_vertices, INF));

    for (int i = 0; i < num_vertices; ++i) {
        dist[i][i] = 0;
    }

    for (const auto& edge : edges) {
        int u = std::get<0>(edge);
        int v = std::get<1>(edge);
        double weight = std::get<2>(edge);
        if (u >= 0 && u < num_vertices && v >= 0 && v < num_vertices) {
            dist[u][v] = weight;
            dist[v][u] = weight; // 无向图
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Floyd-Warshall 算法
    for (int k = 0; k < num_vertices; ++k) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_vertices; ++i) {
            for (int j = 0; j < num_vertices; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_ms = end_time - start_time;

    std::cout << "Time taken for Floyd-Warshall: " << std::fixed << std::setprecision(3) << elapsed_time_ms.count() << " ms" << std::endl;
    std::cout << "Number of threads used: " << num_threads << std::endl;

    std::vector<std::pair<int, int>> test_pairs;
    if (!read_test_cases(test_cases_filename, test_pairs)) {
        return 1;
    }

    std::cout << "\nShortest distances for test pairs:" << std::endl;
    for (const auto& pair : test_pairs) {
        int u = pair.first;
        int v = pair.second;
        if (u >= 0 && u < num_vertices && v >= 0 && v < num_vertices) {
            std::cout << "Shortest distance between " << u << " and " << v << ": ";
            if (dist[u][v] == INF) {
                std::cout << "INF" << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(6) << dist[u][v] << std::endl;
            }
        } else {
            std::cerr << "Warning: Invalid test pair (" << u << ", " << v << ") for the given graph size." << std::endl;
        }
    }

    return 0;
} 