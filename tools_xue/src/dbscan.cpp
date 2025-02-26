#include "dbscan.h"
#include "utils.h"

void dbscan_kernel_wrapper(int b, int n, int c, float eps, int min_point, const float* dataset, int* label);

at::Tensor dbscan_cuda(at::Tensor points, float eps, int min_point) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);


  at::Tensor output =
      torch::full({points.size(0), points.size(1)}, -1,
                  at::device(points.device()).dtype(at::ScalarType::Int));
  



  if (points.is_cuda()) {
    dbscan_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), eps, min_point, points.data_ptr<float>(),
        output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
