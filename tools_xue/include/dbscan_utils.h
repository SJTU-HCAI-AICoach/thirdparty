#pragma once
#include <thrust/device_vector.h>
enum class NodeType : int { CORE, NOISE };

struct Node {
    __host__ __device__ Node()
        : type(NodeType::NOISE)
        , numNeighbors(0)
        , visited(false)
    {
    }

    NodeType type;
    int numNeighbors;
    char visited;
};

struct Graph {
    thrust::device_vector<Node> nodes;
    thrust::device_vector<int> neighborStartIndices;
    thrust::device_vector<int> adjList;
};