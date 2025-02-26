#include "sampling.h"
#include "utils.h"

void furthest_point_sampling_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

void furthest_point_sampling_xue_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, const bool* mask, float *temp,
                                            int *idxs);

void generate_gaussian_heatmap_wrapper(int b, int n, int heatmap_size,
                                            float sigma,
                                            const float *points,
                                            float *output);

at::Tensor generate_gaussian_heatmap(const int heatmap_size, at::Tensor points, const float sigma){
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);
  // points (b, 24, 2)
  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), heatmap_size, heatmap_size},
                   at::device(points.device()).dtype(at::ScalarType::Float));
  if (points.is_cuda()) {
    generate_gaussian_heatmap_wrapper(
        points.size(0), points.size(1), heatmap_size, sigma, points.data_ptr<float>(),
        output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }
  return output;

  
}




at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor furthest_point_sampling_with_mask(at::Tensor points, at::Tensor mask, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(mask);
  CHECK_IS_FLOAT(points);
  CHECK_IS_BOOL(mask);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    furthest_point_sampling_xue_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), nsamples, points.data_ptr<float>(),
        mask.data_ptr<bool>(),tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
