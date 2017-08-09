#include <algorithm>
#include <vector>

#include "caffe/layers/round_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RoundForwardBackward(const int n, const Dtype lower_lim,
    const Dtype upper_lim, const Dtype epsilon, const Dtype* in, Dtype* out, int* under, int* over) {
  under[0] = 0;
  over[0] = 0;
  CUDA_KERNEL_LOOP(index, n) {
    if (in[index] <= lower_lim) {
      out[index] = lower_lim;
      under[0]++;
    } else if (in[index] >= upper_lim) {
      out[index] = upper_lim;
      over[0]++;
    } else {
      Dtype remainder = in[index] - floor(in[index]/epsilon)*epsilon;
      if (remainder < epsilon/2) {
        out[index] = in[index] - remainder;
      } else {
        out[index] = in[index] - remainder + epsilon;
      }
    }
  }
}

template <typename Dtype>
void RoundLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  // Fixed point
  // constant values for fixed point
  int* d_under;
  int* d_over;
  cudaMalloc(&d_under, sizeof(int));
  cudaMalloc(&d_over, sizeof(int));
  if (this->layer_param_.round_param().bypass() == false) {
    RoundForwardBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, lower_lim_, upper_lim_, epsilon_, bottom_data, top_data, d_under, d_over);
    CUDA_POST_KERNEL_CHECK;
    cudaMemcpy(&data_under_, d_under, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&data_over_, d_over, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_under);
    cudaFree(d_over);
    //LOG(INFO) << "GPU round forward under: " << h_under << " over: " << h_over;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void RoundLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // Fixed point
    // constant values for fixed point
    int* d_under;
    int* d_over;
    cudaMalloc(&d_under, sizeof(int));
    cudaMalloc(&d_over, sizeof(int));
    if (this->layer_param_.round_param().bypass() == false) {
      RoundForwardBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, lower_lim_, upper_lim_, epsilon_, top_diff, bottom_diff, d_under, d_over);
      CUDA_POST_KERNEL_CHECK;
      cudaMemcpy(&diff_under_, d_under, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&diff_over_, d_over, sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(d_under);
      cudaFree(d_over);
      //LOG(INFO) << "GPU round backward under: " << h_under << " over: " << h_over;
    } else {
      caffe_copy(count, top_diff, bottom_diff);
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(RoundLayer);


}  // namespace caffe
