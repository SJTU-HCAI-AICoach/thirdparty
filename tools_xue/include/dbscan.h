#pragma once
#include <torch/extension.h>

at::Tensor dbscan_cuda(at::Tensor points, float eps, int min_point);
