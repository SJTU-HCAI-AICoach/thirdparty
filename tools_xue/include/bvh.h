#pragma once
#include <torch/extension.h>

std::vector<torch::Tensor> bvh_ray_tracing(torch::Tensor triangles,
                                           torch::Tensor points,
                                           torch::Tensor directions,
                                           int queue_size,
                                           bool sort_points_by_morton);