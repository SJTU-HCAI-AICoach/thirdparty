#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2)
{
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int c, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs)
{
  if (m <= 0)
    return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * c;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0)
    idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++)
  {
    int besti = 0;
    float best = -1;
    // float x1 = dataset[old * 3 + 0];
    // float y1 = dataset[old * 3 + 1];
    // float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride)
    {
      float dis, d = 0.0f;
      for (int ci = 0; ci < c; ci++){
        dis = dataset[old * c + ci] - dataset[k * c + ci];
        d += dis * dis;
      }
      // float x2, y2, z2;
      // x2 = dataset[k * 3 + 0];
      // y2 = dataset[k * 3 + 1];
      // z2 = dataset[k * 3 + 2];
      // float d = 0.0;
      // for (int idx = 0; idx < c; idx += 1){
      //   float x1 = dataset[old * c + idx];
      //   float x2 = dataset[k * c + idx];
      //   d += (x2 - x1) * (x2 - x1);
      // }
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3) continue;

      // float d =
      //     (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512)
    {
      if (tid < 256)
      {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256)
    {
      if (tid < 128)
      {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128)
    {
      if (tid < 64)
      {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64)
    {
      if (tid < 32)
      {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32)
    {
      if (tid < 16)
      {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16)
    {
      if (tid < 8)
      {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8)
    {
      if (tid < 4)
      {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4)
    {
      if (tid < 2)
      {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2)
    {
      if (tid < 1)
      {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
      idxs[j] = old;
  }
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_xue_kernel(
    int b, int n, int c, int m,
    const float *__restrict__ dataset,
    const bool *__restrict__ mask,
    float *__restrict__ temp,
    int *__restrict__ idxs)
{
  if (m <= 0)
    return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * c;
  temp += batch_index * n;
  idxs += batch_index * m;
  mask += batch_index * n;

  int tid = threadIdx.x;
  const int stride = block_size;
  int old = -1;  
  for (int k = 0; k < n; ++k) {  
    if (mask[k]) {  
        old = k;  
        break;  
    }  
  }
  if (old == -1) {  
    printf("Error: No valid starting point found!\n");  
    return;  
  }
  if(threadIdx.x == 0){
    idxs[0]=old;
  }
  __syncthreads();
  // TODO resampling when given points are less than requested points
  // int n_valid = 0;
  // for(int k=0;k<n;++k) if(mask[k]) n_valid++;
  // int m_selected = 1;
  for (int j = 1; j < m ; j++)
  {
    int besti = 0;
    float best = -1;
    // float x1 = dataset[old * 3 + 0];
    // float y1 = dataset[old * 3 + 1];
    // float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride)
    {
      if (!mask[k]) continue;
      float dis, d = 0.0f;
      for (int ci = 0; ci < c; ci++){
        dis = dataset[old * c + ci] - dataset[k * c + ci];
        d += dis * dis;
      }
      // float x2, y2, z2;
      // x2 = dataset[k * 3 + 0];
      // y2 = dataset[k * 3 + 1];
      // z2 = dataset[k * 3 + 2];
      // float d = 0.0;
      // for (int idx = 0; idx < c; idx += 1){
      //   float x1 = dataset[old * c + idx];
      //   float x2 = dataset[k * c + idx];
      //   d += (x2 - x1) * (x2 - x1);
      // }
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3) continue;

      // float d =
      //     (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512)
    {
      if (tid < 256)
      {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256)
    {
      if (tid < 128)
      {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128)
    {
      if (tid < 64)
      {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64)
    {
      if (tid < 32)
      {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32)
    {
      if (tid < 16)
      {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16)
    {
      if (tid < 8)
      {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8)
    {
      if (tid < 4)
      {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4)
    {
      if (tid < 2)
      {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2)
    {
      if (tid < 1)
      {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
      idxs[j] = old;
  }
}
void furthest_point_sampling_xue_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, const bool* mask, float *temp,
                                            int *idxs)
{
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads)
  {
  case 512:
    furthest_point_sampling_xue_kernel<512>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 256:
    furthest_point_sampling_xue_kernel<256>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 128:
    furthest_point_sampling_xue_kernel<128>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 64:
    furthest_point_sampling_xue_kernel<64>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 32:
    furthest_point_sampling_xue_kernel<32>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 16:
    furthest_point_sampling_xue_kernel<16>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 8:
    furthest_point_sampling_xue_kernel<8>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 4:
    furthest_point_sampling_xue_kernel<4>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 2:
    furthest_point_sampling_xue_kernel<2>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  case 1:
    furthest_point_sampling_xue_kernel<1>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
    break;
  default:
    furthest_point_sampling_xue_kernel<512>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset,mask, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int c, int m,
                                            const float *dataset, float *temp,
                                            int *idxs)
{
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads)
  {
  case 512:
    furthest_point_sampling_kernel<512>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 256:
    furthest_point_sampling_kernel<256>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 128:
    furthest_point_sampling_kernel<128>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 64:
    furthest_point_sampling_kernel<64>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 32:
    furthest_point_sampling_kernel<32>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 16:
    furthest_point_sampling_kernel<16>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 8:
    furthest_point_sampling_kernel<8>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 4:
    furthest_point_sampling_kernel<4>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 2:
    furthest_point_sampling_kernel<2>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  case 1:
    furthest_point_sampling_kernel<1>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
    break;
  default:
    furthest_point_sampling_kernel<512>
        <<<b, n_threads, 0, stream>>>(b, n, c, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}



__global__ void generate_gaussian_heatmap_kernel(
    int b, int n, int heatmap_size, float sigma,
    const float *__restrict__ points,
    float *__restrict__ output)
{
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx >= b * n * heatmap_size * heatmap_size) {
      return;
  }

  int batch_idx = idx / (n * heatmap_size * heatmap_size);
  int point_idx = (idx / (heatmap_size * heatmap_size)) % n;

  int out_y = (idx / heatmap_size) % heatmap_size;
  int out_x = idx % heatmap_size;

  float point_x = points[batch_idx * n * 2 + point_idx * 2 + 0];
  float point_y = points[batch_idx * n * 2 + point_idx * 2 + 1];

  float exponent = -(pow(out_x - point_x, 2) + pow(out_y - point_y, 2)) / (1 * pow(sigma, 2));//(2 * pow(sigma, 2));

  if (exponent < -87.0f) {
      output[idx] = 0.0f;
  } else {
      output[idx] = exp(exponent);//(2 * M_PI * sigma * sigma);
  }
}
void generate_gaussian_heatmap_wrapper(int b, int n, int heatmap_size,float sigma,
  const float *points,
  float *output){
    unsigned int n_threads = opt_n_threads(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 block_size(n_threads);
    dim3 grid_size((b * n * heatmap_size * heatmap_size + block_size.x - 1) / block_size.x);
    generate_gaussian_heatmap_kernel<<<grid_size, block_size, 0, stream>>>(b, n, heatmap_size, sigma, points, output);
  
    CUDA_CHECK_ERRORS();
    
  }
