#pragma once
#include <torch/extension.h>

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
at::Tensor furthest_point_sampling_with_mask(at::Tensor points, at::Tensor mask, const int nsamples);
at::Tensor generate_gaussian_heatmap(const int heatmap_size, at::Tensor points, const float sigma);
