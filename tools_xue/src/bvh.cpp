#include <torch/extension.h>

#include <iostream>
#include <limits>
#include <vector>

void bvh_ray_tracing_kernel(const torch::Tensor& triangles,
                            const torch::Tensor& points,
                            const torch::Tensor& directions,
                            torch::Tensor* distances,
                            torch::Tensor* closest_points,
                            torch::Tensor* closest_faces,
                            torch::Tensor* closest_bcs,
                            int queue_size = 128,
                            bool sort_points_by_morton = true);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bvh_ray_tracing(torch::Tensor triangles,
                                           torch::Tensor points,
                                           torch::Tensor directions,
                                           int queue_size = 128,
                                           bool sort_points_by_morton = true) {
    CHECK_INPUT(triangles);
    CHECK_INPUT(points);
    CHECK_INPUT(directions);

    auto options = torch::TensorOptions()
                       .dtype(triangles.dtype())
                       .layout(triangles.layout())
                       .device(triangles.device());

    torch::Tensor distances = torch::full({triangles.size(0), points.size(1)}, -1, options);
    torch::Tensor closest_points = torch::full({triangles.size(0), points.size(1), 3}, -1, options);
    torch::Tensor closest_bcs = torch::full({triangles.size(0), points.size(1), 3}, -1, options);
    torch::Tensor closest_faces = torch::full({triangles.size(0), points.size(1)}, -1, torch::TensorOptions().dtype(torch::kLong).layout(triangles.layout()).device(triangles.device()));

    bvh_ray_tracing_kernel(triangles, points, directions,
                           &distances, &closest_points, &closest_faces,
                           &closest_bcs,
                           queue_size, sort_points_by_morton);

    return {distances, closest_points, closest_faces, closest_bcs};
}

