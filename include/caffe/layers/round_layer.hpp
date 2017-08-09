#ifndef CAFFE_ROUND_LAYER_HPP_
#define CAFFE_ROUND_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Round unit for quantization @f$ y = round(x, <IL,FL>) @f$.
 */
template <typename Dtype>
class RoundLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides RoundParameter round_param,
   *     with RoundLayer options:
   *   - i_length (\b optional, default 0).
   *     the value @f$ \nu @f$ interger part of the number.
   *   - f_length (\b optional, default 0).
   *     the value @f$ \nu @f$ fractional part of the number.
   */
  explicit RoundLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  inline void set_flength(const int value) {
    flength_ = value;
  }
  inline void set_ilength(const int value) {
    ilength_ = value;
  }
  inline void set_amean(const Dtype value) {
    amean_ = value;
  }
  inline void set_astd(const Dtype value) {
    astd_ = value;
  }
  inline void set_epsilon() {
    epsilon_ = pow(2, -flength_);
    upper_lim_ = pow(2, ilength_) - epsilon_;
    lower_lim_ = -pow(2, ilength_);
  }
  inline void set_bypass(const bool value) {
    bypass_ = value;
  }
  inline int flength() const { return flength_; }
  inline int ilength() const { return ilength_; }
  inline int data_under() const { return data_under_; }
  inline int data_over() const { return data_over_; }
  inline int diff_under() const { return diff_under_; }
  inline int diff_over() const { return diff_over_; }
  inline bool bypass() const { return bypass_; }
  inline Dtype epsilon() const { return epsilon_; }
  inline Dtype upper_lim() const { return upper_lim_; }
  inline Dtype lower_lim() const { return upper_lim_; }
  inline Dtype amean() const { return amean_; }
  inline Dtype astd() const { return astd_; }

  virtual inline const char* type() const { return "Round"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = round(x, <IL,FL>)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the Round inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            0 & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$ if propagate_down[0], by default.
   *      If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed gradients are @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            \nu \frac{\partial E}{\partial y} & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int data_under_;
  int data_over_;
  int diff_under_;
  int diff_over_;
  int flength_;
  int ilength_;
  bool bypass_;
  Dtype epsilon_;
  Dtype upper_lim_;
  Dtype lower_lim_;
  Dtype amean_;
  Dtype astd_;
};

}  // namespace caffe

#endif  // CAFFE_ROUND_LAYER_HPP_
