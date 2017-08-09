#include <algorithm>
#include <vector>

#include "caffe/layers/round_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RoundLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  data_under_ = 0;
  data_over_ = 0;
  diff_under_ = 0;
  diff_over_ = 0;
  amean_ = 0.0;
  astd_ = 0.0;
  flength_ = this->layer_param_.round_param().f_length();
  ilength_ = this->layer_param_.round_param().i_length();
  epsilon_ = pow(2, -flength_);
  upper_lim_ = pow(2, ilength_) - epsilon_;
  lower_lim_ = -pow(2, ilength_);
  bypass_ = this->layer_param_.round_param().bypass();
}

template <typename Dtype>
void RoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  // Fixed point
  // constant values for fixed point
  data_under_ = 0;
  data_over_ = 0;
  if (bypass_ == false) {
    caffe_round(count,
        lower_lim_, upper_lim_, epsilon_,
        bottom_data, top_data,
        &data_under_, &data_over_);
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void RoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    // Fixed point
    // constant values for fixed point
    diff_under_ = 0;
    diff_over_ = 0;
    if (bypass_ == false) {
      caffe_round(count,
          lower_lim_, upper_lim_, epsilon_,
          top_diff, bottom_diff,
          &diff_under_, &diff_over_);
    } else {
      caffe_copy(count, top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(RoundLayer);
#endif

INSTANTIATE_CLASS(RoundLayer);

}  // namespace caffe
