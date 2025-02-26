#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>  
#include "cuda_utils.h"
#include "dbscan_utils.h"
#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */
// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                             \
    {                                                                \
        cudaDeviceSynchronize();                                     \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess) {                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(0);                                                 \
        }                                                            \
    }
#else
#define cudaCheckError()
#endif

__global__ void makeGraphStep1Kernel(const float* __restrict__ points, Node* nodes, int* nodeDegs, int numPoints,
                                     float eps, int minPoints)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < numPoints) {
        Node* node = &nodes[tid];
        const float* point = points + tid*3;

        for (int i = 0; i < numPoints; ++i) {
            if (i == tid) {
                continue;
            }

            const float* curPoint = points + i*3;
            float diffx = point[0] - curPoint[0];
            float diffy = point[1] - curPoint[1];
            float diffz = point[2] - curPoint[2];

            float sqrDist = diffx * diffx + diffy * diffy + diffz * diffz;


            if (sqrDist < eps * eps) {
                node->numNeighbors++;
            }
        }

        if (node->numNeighbors >= minPoints) {
            node->type = NodeType::CORE;
        }

        nodeDegs[tid] = node->numNeighbors;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void makeGraphStep2Kernel(const float* __restrict__ points, const int* __restrict__ neighborStartIndices,
                                     int* adjList, int numPoints, float eps)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // points (n,3)

    while (tid < numPoints) {
        const float* point = points + tid*3;
        int startIdx = neighborStartIndices[tid];

        int countNeighbors = 0;
        for (int i = 0; i < numPoints; ++i) {
            if (i == tid) {
                continue;
            }

            const float* curPoint = points + i*3;
            float diffx = point[0]- curPoint[0];
            float diffy = point[1] - curPoint[1];
            float diffz = point[2] - curPoint[2];

            float sqrDist = diffx * diffx + diffy * diffy + diffz * diffz;

            if (sqrDist < eps * eps) {
                adjList[startIdx + countNeighbors++] = i;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void BFSKernel(const Node* __restrict__ nodes, const int* __restrict__ adjList,
                          const int* __restrict__ neighborStartIndices, char* Fa, char* Xa,  int numPoints, int b, int* out, int current_label)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < numPoints) {
        if (Fa[tid]) {
            Fa[tid] = false;
            Xa[tid] = true;
            out[tid + b * numPoints] = current_label;

            int startIdx = neighborStartIndices[tid];

            for (int i = 0; i < nodes[tid].numNeighbors; ++i) {
                int nIdx = adjList[startIdx + i];
                Fa[nIdx] = 1 - Xa[nIdx];
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}
  

void BFS(Node* hNodes, Node* dNodes, const int* adjList, const int* neighborStartIndices, int v, int numPoints,
         int b, int* out, int current_label)
{
    thrust::device_vector<char> dXa(numPoints, false);
    thrust::device_vector<char> dFa(numPoints, false);
    dFa[v] = true;

    int numThreads = 256;
    int numBlocks = std::min(512, (numPoints + numThreads - 1) / numThreads);

    int countFa = 1;
    while (countFa > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        BFSKernel<<<numBlocks, numThreads, 0, stream>>>(dNodes, adjList, neighborStartIndices,
                                             thrust::raw_pointer_cast(dFa.data()), thrust::raw_pointer_cast(dXa.data()),
                                             numPoints, b, out, current_label);
        countFa = thrust::count(thrust::device, dFa.begin(), dFa.end(), true);
        cudaCheckError();
    }

    thrust::host_vector<char> hXa = dXa;
    // cudaCheckError();

    for (int i = 0; i < numPoints; ++i) {
        if (hXa[i]) {
            hNodes[i].visited = true;
            // curCluster.emplace_back(i);
        }
    }
}

Graph makeGraph(const float* points, int numPoints, float eps, int minPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    Graph graph;

    graph.nodes.resize(numPoints);
    graph.neighborStartIndices.resize(numPoints);

    thrust::device_vector<int> dNodeDegs(numPoints);

    int numThreads = 256;
    int numBlocks = std::min(512, (numPoints + numThreads - 1) / numThreads);

    makeGraphStep1Kernel<<<numBlocks, numThreads, 0, stream>>>(points, thrust::raw_pointer_cast(graph.nodes.data()),
                                                               thrust::raw_pointer_cast(dNodeDegs.data()), numPoints,
                                                               eps, minPoints);

    thrust::exclusive_scan(dNodeDegs.begin(), dNodeDegs.end(), graph.neighborStartIndices.begin());

    int totalEdges = dNodeDegs.back() + graph.neighborStartIndices.back();
    graph.adjList.resize(totalEdges);

    makeGraphStep2Kernel
        <<<numBlocks, numThreads, 0, stream>>>(points, thrust::raw_pointer_cast(graph.neighborStartIndices.data()),
                                    thrust::raw_pointer_cast(graph.adjList.data()), numPoints, eps);

    cudaCheckError();
    return graph;
}

void dbscan_kernel_wrapper(int b, int n, int c, float eps, int min_points, const float *dataset, int *idxs)
{
  for(int i=0;i<b;++i){
    const float* data = dataset + i * n * c;
    auto graph = makeGraph(data, n, eps, min_points);
    thrust::host_vector<Node> hNodes = graph.nodes;
    int current_label = 0;
    for (int j = 0; j < n; ++j) {
        auto& curHNode = hNodes[j];
        if (curHNode.visited || curHNode.type != NodeType::CORE) {
            continue;
        }
        // std::vector<int> curCluster;
        // curCluster.emplace_back(j);
        curHNode.visited = true;
        BFS(hNodes.data(), thrust::raw_pointer_cast(graph.nodes.data()), thrust::raw_pointer_cast(graph.adjList.data()),
            thrust::raw_pointer_cast(graph.neighborStartIndices.data()), j, n, i, idxs, current_label);
        current_label += 1;
    }
  }
}

  