#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::ComputeStatsRound(int round_layer_id) {
  int layer_id = this->net_->round_layer_ids()[round_layer_id];
  const vector<Blob<Dtype>*>& round_top = this->net_->top_vecs()[layer_id];

  Dtype amean_data = 0;
  Dtype amax_data = 0;
  int   iamax = 0;
  Dtype meansq_data = 0;
  Dtype astd_data = 0;
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    iamax = caffe_cpu_iamax(round_top[0]->count(),
                            round_top[0]->cpu_data());
    amax_data = std::fabs(round_top[0]->cpu_data()[iamax]);
    amean_data = caffe_cpu_asum(round_top[0]->count(),
                           round_top[0]->cpu_data());
    meansq_data = caffe_cpu_dot(round_top[0]->count(),
                            round_top[0]->cpu_data(),
                            round_top[0]->cpu_data());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_iamax(round_top[0]->count(),
                   round_top[0]->gpu_data(), &iamax);
    amax_data = std::fabs(round_top[0]->cpu_data()[iamax]);
    caffe_gpu_asum(round_top[0]->count(),
                   round_top[0]->gpu_data(), &amean_data);
    caffe_gpu_dot(round_top[0]->count(),
              round_top[0]->gpu_data(),
              round_top[0]->gpu_data(),
              &meansq_data);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  amean_data /= round_top[0]->count();
  meansq_data /= round_top[0]->count();
  astd_data = sqrt(meansq_data - amean_data*amean_data);
  this->net_->set_round_layers_amean_data(round_layer_id, amean_data);
  this->net_->set_round_layers_amax_data(round_layer_id, amax_data);
  this->net_->set_round_layers_astd_data(round_layer_id, astd_data);

  // Compute diff statistics
  Dtype amean_diff = round_top[0]->asum_diff()/round_top[0]->count();
  Dtype meansq_diff = round_top[0]->sumsq_diff()/round_top[0]->count();
  Dtype astd_diff = sqrt(meansq_diff - amean_diff*amean_diff);
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    iamax = caffe_cpu_iamax(round_top[0]->count(),
                            round_top[0]->cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_iamax(round_top[0]->count(),
                   round_top[0]->gpu_diff(), &iamax);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  Dtype amax_diff = std::fabs(round_top[0]->cpu_diff()[iamax]);
  this->net_->set_round_layers_amean_diff(round_layer_id, amean_diff);
  this->net_->set_round_layers_amax_diff(round_layer_id, amax_diff);
  this->net_->set_round_layers_astd_diff(round_layer_id, astd_diff);


  //LOG(INFO) << this->net_->layer_names()[layer_id]
  //          << " Top data, " << round_top[0]->shape_string()
  //          << ", Absolute mean: " << amean_data << ", Std: " << astd_data << ", Amax: " << amax_data;

}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::ComputeStatsParam(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype amean_data = 0;
  Dtype amax_data = 0;
  int   iamax = 0;
  Dtype meansq_data = 0;
  Dtype astd_data = 0;
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    iamax = caffe_cpu_iamax(net_params[param_id]->count(),
                            net_params[param_id]->cpu_data());
    amax_data = std::fabs(net_params[param_id]->cpu_data()[iamax]);
    amean_data = caffe_cpu_asum(net_params[param_id]->count(),
                           net_params[param_id]->cpu_data());
    meansq_data = caffe_cpu_dot(net_params[param_id]->count(),
                            net_params[param_id]->cpu_data(),
                            net_params[param_id]->cpu_data());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_iamax(net_params[param_id]->count(),
                    net_params[param_id]->gpu_data(), &iamax);
    amax_data = std::fabs(net_params[param_id]->cpu_data()[iamax]);
    caffe_gpu_asum(net_params[param_id]->count(),
                   net_params[param_id]->gpu_data(), &amean_data);
    caffe_gpu_dot(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              net_params[param_id]->gpu_data(),
              &meansq_data);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  amean_data /= net_params[param_id]->count();
  meansq_data /= net_params[param_id]->count();
  astd_data = sqrt(meansq_data - amean_data*amean_data);
  this->net_->set_params_amean_data(param_id, amean_data);
  this->net_->set_params_amax_data(param_id, amax_data);
  this->net_->set_params_astd_data(param_id, astd_data);

  // Compute diff statistics
  Dtype amean_diff = net_params[param_id]->asum_diff()/net_params[param_id]->count();
  Dtype meansq_diff = net_params[param_id]->sumsq_diff()/net_params[param_id]->count();
  Dtype astd_diff = sqrt(meansq_diff - amean_diff*amean_diff);
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    iamax = caffe_cpu_iamax(net_params[param_id]->count(),
                            net_params[param_id]->cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_iamax(net_params[param_id]->count(),
                    net_params[param_id]->gpu_diff(), &iamax);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  Dtype amax_diff = std::fabs(net_params[param_id]->cpu_diff()[iamax]);
  this->net_->set_params_amean_diff(param_id, amean_diff);
  this->net_->set_params_amax_diff(param_id, amax_diff);
  this->net_->set_params_astd_diff(param_id, astd_diff);

  //int layer_id = this->net_->learnable_param_layer_ids()[param_id];
  //LOG(INFO) << this->net_->layer_names()[layer_id]
  //          << ", " << net_params[param_id]->shape_string()
  //          << ", Absolute mean: " << amean_data << ", Std: " << astd_data << ", Amax: " << amax_data;
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();

  // tna6 added
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  // tna6
  PrintParamsStats();

  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();

  // tna6 added
  // Fixed
  if (this->param_.solver_dtype() == caffe::SolverParameter_SolverDtype_FIXED) {
    for (int param_id = 0; param_id < net_params.size();
         ++param_id) {
      Dtype epsilon = this->net_->params_epsilon()[param_id];
      Dtype upper_lim = this->net_->params_upperlim()[param_id];
      Dtype lower_lim = this->net_->params_lowerlim()[param_id];
      Round(param_id, lower_lim, upper_lim, epsilon);
    }
    if (this->param_.solver_fixed_mode() == caffe::SolverParameter_SolverFixedMode_DYNAMIC) {
      DynamicPrecisionControl();
    } // end dynamic
  } // end fixed
}

// tna6
template <typename Dtype>
void SGDSolver<Dtype>::PrintParamsStats() {

  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<RoundLayer<Dtype>*>& net_round_layers = this->net_->round_layers();

  if (this->param_.params_display() && this->iter_ % this->param_.params_display() == 0) {
    if (this->param_.solver_dtype() == caffe::SolverParameter_SolverDtype_FIXED) {
      LOG(INFO) << "Iteration " << this->iter_ << ", Parameters information";
      for (int param_id = 0; param_id < net_params.size();
           ++param_id) {
        int layer_id = this->net_->learnable_param_layer_ids()[param_id];
        int flength = this->net_->params_flength()[param_id];
        int ilength = this->net_->params_ilength()[param_id];
        Dtype epsilon = this->net_->params_epsilon()[param_id];
        Dtype upper_lim = this->net_->params_upperlim()[param_id];
        Dtype lower_lim = this->net_->params_lowerlim()[param_id];
        LOG(INFO) << this->net_->layer_names()[layer_id]
                  << ", " << net_params[param_id]->shape_string()
                  << ", total length: " << flength+ilength
                  << ", i: " << ilength
                  << ", f: " << flength << ", epsilon: " << epsilon
                  << ", upper_lim: " << upper_lim << ", lower_lim: " << lower_lim;
      }
      LOG(INFO) << "Iteration " << this->iter_ << ", Round layers information";
      for (int round_layer_id = 0; round_layer_id < net_round_layers.size();
           ++round_layer_id) {
        int flength = net_round_layers[round_layer_id]->flength();
        int ilength = net_round_layers[round_layer_id]->ilength();
        Dtype epsilon   = net_round_layers[round_layer_id]->epsilon();
        Dtype upper_lim = net_round_layers[round_layer_id]->upper_lim();
        Dtype lower_lim = net_round_layers[round_layer_id]->lower_lim();
        LOG(INFO) << net_round_layers[round_layer_id]->layer_param().name()
                  << ", total length: " << flength+ilength
                  << ", i: " << ilength
                  << ", f: " << flength << ", epsilon: " << epsilon
                  << ", upper_lim: " << upper_lim << ", lower_lim: " << lower_lim;
      }
    }
    // print stats
    LOG(INFO) << "Iteration " << this->iter_ << ", Parameters information";
    for (int param_id = 0; param_id < net_params.size();
         ++param_id) {
      ComputeStatsParam(param_id);
      int layer_id = this->net_->learnable_param_layer_ids()[param_id];
      Dtype amean_data = this->net_->params_amean_data()[param_id];
      Dtype astd_data = this->net_->params_astd_data()[param_id];
      Dtype amax_data = this->net_->params_amax_data()[param_id];
      Dtype amean_diff = this->net_->params_amean_diff()[param_id];
      Dtype astd_diff = this->net_->params_astd_diff()[param_id];
      Dtype amax_diff = this->net_->params_amax_diff()[param_id];
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << ", " << net_params[param_id]->shape_string()
                << ", amean_data: " << amean_data << ", astd_data: " << astd_data << ", amax_data: " << amax_data;
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << ", " << net_params[param_id]->shape_string()
                << ", amean_diff: " << amean_diff << ", astd_diff: " << astd_diff << ", amax_diff: " << amax_diff;
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << ", " << net_params[param_id]->shape_string()
                << ", amean_diff/amean_data: " << amean_diff/amean_data;
    }
    LOG(INFO) << "Iteration " << this->iter_ << ", Round layers information";
    for (int round_layer_id = 0; round_layer_id < net_round_layers.size();
         ++round_layer_id) {
      ComputeStatsRound(round_layer_id);
      int layer_id = this->net_->round_layer_ids()[round_layer_id];
      Dtype amean_data = this->net_->round_layers_amean_data()[round_layer_id];
      Dtype astd_data = this->net_->round_layers_astd_data()[round_layer_id];
      Dtype amax_data = this->net_->round_layers_amax_data()[round_layer_id];
      Dtype amean_diff = this->net_->round_layers_amean_diff()[round_layer_id];
      Dtype astd_diff = this->net_->round_layers_astd_diff()[round_layer_id];
      Dtype amax_diff = this->net_->round_layers_amax_diff()[round_layer_id];
      const vector<Blob<Dtype>*>& round_top = this->net_->top_vecs()[layer_id];
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << " Top data, " << round_top[0]->shape_string()
                << ", amean_data: " << amean_data << ", astd_data: " << astd_data << ", amax_data: " << amax_data;
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << " Top diff, " << round_top[0]->shape_string()
                << ", amean_diff: " << amean_diff << ", astd_diff: " << astd_diff << ", amax_diff: " << amax_diff;
      LOG(INFO) << this->net_->layer_names()[layer_id]
                << " Top diff, " << round_top[0]->shape_string()
                << ", amean_diff/amean_data: " << amean_diff/amean_data;
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  const bool display = this->param_.display() && this->iter_ % this->param_.display() == 0;

  // tna6 added
  // Fixed point
  // constant values for fixed point
  Dtype zero_out = this->param_.zero_out();
  const Dtype epsilon = this->net_->params_epsilon()[param_id];
  const Dtype upper_lim = this->net_->params_upperlim()[param_id];
  const Dtype lower_lim = this->net_->params_lowerlim()[param_id];
  const bool fixed = this->param_.solver_dtype() == caffe::SolverParameter_SolverDtype_FIXED;

  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {

    // tna6
    if (zero_out > 0) {
      int num_zero = 0;
      caffe_zero_out_inplace(net_params[param_id]->count(),
          rate * zero_out,
          net_params[param_id]->mutable_cpu_diff(),
          &num_zero);
      //caffe_zero_out_inplace(net_params[param_id]->count(),
      //    local_rate * zero_out,
      //    history_[param_id]->mutable_cpu_data(),
      //    &num_zero);
      if (display) {
        LOG(INFO)   << "local_rate: " << local_rate
                    << ", zero_out: " << rate * zero_out
                    << ", history: " << history_[param_id]->mutable_cpu_data()[0]
                    << ", # zeroing out: " << num_zero
                    << " out of: " << net_params[param_id]->count();
      }
    }

    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());

    // tna6
    if (fixed) {
      int under = 0;
      int over = 0;
      caffe_round_inplace(net_params[param_id]->count(),
          lower_lim, upper_lim, epsilon,
          history_[param_id]->mutable_cpu_data(),
          &under, &over);
      this->net_->set_params_diff_under(param_id, under);
      this->net_->set_params_diff_over(param_id, over);
      //LOG(INFO) << "under: " << under << ", over: " << over;
    }

    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // tna6
//    sgd_update_gpu(net_params[param_id]->count(),
//        net_params[param_id]->mutable_gpu_diff(),
//        history_[param_id]->mutable_gpu_data(),
//        momentum, local_rate);
    // tna6
    // Zeroing out
    if (zero_out > 0) {
      int h_num_zero;
      int* d_num_zero;
      cudaMalloc(&d_num_zero, sizeof(int));
      caffe_gpu_zero_out_inplace(net_params[param_id]->count(),
          rate * zero_out,
          net_params[param_id]->mutable_gpu_diff(),
          d_num_zero);
      //caffe_gpu_zero_out_inplace(net_params[param_id]->count(),
      //    local_rate * zero_out,
      //    history_[param_id]->mutable_gpu_data(),
      //    d_num_zero);
      cudaMemcpy(&h_num_zero, d_num_zero, sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(d_num_zero);
      if (display) {
        LOG(INFO)   << "local_rate: " << local_rate
                    << ", zero_out: " << rate * zero_out
                    << ", # zeroing out: " << h_num_zero
                    << " out of: " << net_params[param_id]->count();
      }
    }

    caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->gpu_diff(), momentum,
              history_[param_id]->mutable_gpu_data());

    // tna6
    if (fixed) {
      int h_under;
      int* d_under;
      int h_over;
      int* d_over;
      cudaMalloc(&d_under, sizeof(int));
      cudaMalloc(&d_over, sizeof(int));
      caffe_gpu_round_inplace(net_params[param_id]->count(),
          lower_lim, upper_lim, epsilon,
          history_[param_id]->mutable_gpu_data(),
          d_under, d_over);
      cudaMemcpy(&h_under, d_under, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_over, d_over, sizeof(int), cudaMemcpyDeviceToHost);
      cudaFree(d_under);
      cudaFree(d_over);
      this->net_->set_params_diff_under(param_id, h_under);
      this->net_->set_params_diff_over(param_id, h_over);
      //LOG(INFO) << "under: " << h_under << ", over: " << h_over;
    }

    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->gpu_data(),
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::Round(int param_id, Dtype lower_lim, Dtype upper_lim, Dtype epsilon) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    int under = 0;
    int over = 0;
    caffe_round_inplace(net_params[param_id]->count(),
        lower_lim, upper_lim, epsilon,
        net_params[param_id]->mutable_cpu_data(),
        &under, &over);
    this->net_->set_params_under(param_id, under);
    this->net_->set_params_over(param_id, over);
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    int h_under;
    int* d_under;
    int h_over;
    int* d_over;
    cudaMalloc(&d_under, sizeof(int));
    cudaMalloc(&d_over, sizeof(int));
    caffe_gpu_round_inplace(net_params[param_id]->count(),
        lower_lim, upper_lim, epsilon,
        net_params[param_id]->mutable_gpu_data(),
        d_under, d_over);
    cudaMemcpy(&h_under, d_under, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_over, d_over, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_under);
    cudaFree(d_over);
    this->net_->set_params_under(param_id, h_under);
    this->net_->set_params_over(param_id, h_over);
    //LOG(INFO) << "under: " << h_under << ", over: " << h_over;
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::DynamicPrecisionControl() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<RoundLayer<Dtype>*>& net_round_layers = this->net_->round_layers();
  int average_loss = this->param_.average_loss();
  int idx = this->iter_ % average_loss;
  int preidx = (this->iter_-1) % average_loss;
  int ipreidx2 = std::min((int)this->smoothed_losses_.size()-1, (int)this->param_.dsearch_window());
  int preidx2 = (this->iter_-ipreidx2) % average_loss;
  //LOG(INFO) << "Dynamic Precision Control enter ...";
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    ComputeStatsParam(param_id);
  }
  for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
    ComputeStatsRound(round_layer_id);
  }

  // Overflow check
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    DynamicPrecisionParamOverflowCheck(param_id);
  }
  for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
    DynamicPrecisionRoundOverflowCheck(round_layer_id);
  }

  // dynamic_precision_stage_ control
  switch (this->dynamic_precision_stage_) {
  case SolverDynamicPrecision::ISEARCH: {
    int all_fsearch = 0;
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      all_fsearch += DynamicPrecisionParamIsearch(param_id);
    }
    for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
      all_fsearch += DynamicPrecisionRoundIsearch(round_layer_id);
    }
    // if all parameters and round layer blobs integer parts are set,
    // change dynamic_precision_stage_ to DSEARCH
    if (all_fsearch == (net_params.size()+net_round_layers.size())) {
      this->set_dynamic_precision_stage(SolverDynamicPrecision::DSEARCH);
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        int ilength = this->net_->params_ilength()[param_id];
        int flength = this->param_.target_length() - ilength;
        this->net_->set_params_flength(param_id, flength);
        this->net_->set_params_epsilon(param_id);
      }
      for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
        int ilength =    net_round_layers[round_layer_id]->ilength();
        int flength = this->param_.target_length() - ilength;
        net_round_layers[round_layer_id]->set_flength(flength);
        net_round_layers[round_layer_id]->set_epsilon();
      }
      LOG(INFO) << "Iteration " << this->iter_
                << ", [ISEARCH --> DSEARCH] "
                << " total bit width of all learnable params and round layers: " << this->param_.target_length();
    }
    break;
  }
  case SolverDynamicPrecision::DSEARCH: {
    if (this->smoothed_losses_[idx] ==
        *std::max_element(this->smoothed_losses_.begin(),this->smoothed_losses_.end())) { // if increasing
      LOG(INFO) << "Iteration " << this->iter_
                << ", num_loss_increasing: " << this->num_loss_increasing_;
      if(this->param_.dsearch_window() == this->num_loss_increasing_) {
        LOG(INFO) << "[DSEARCH] loss keeps increasing";
        LOG(INFO) << ", previous       loss: " << this->smoothed_losses_[preidx] << " preidx: " << preidx;
        LOG(INFO) << ", current        loss: " << this->smoothed_losses_[idx] << " idx: " << idx;
        LOG(INFO) << ", dsearch window loss: " << this->smoothed_losses_[preidx2] << " preidx2: " << preidx2;
        int ilength = this->net_->params_ilength()[0];
        int flength = this->net_->params_flength()[0];
        int inc_step = this->param_.max_length() - ilength - flength;
        if (ilength+flength < this->param_.max_length()) {
          this->min_smoothed_loss_ =  *std::min_element(this->smoothed_losses_.begin(),this->smoothed_losses_.end());
          if (this->current_target_length_ < this->param_.max_length()-this->param_.bit_step()) {
            this->current_target_length_ = ilength+flength+this->param_.bit_step();
          }
          LOG(INFO) << "[DSEARCH] min_smoothed_loss_: " << this->min_smoothed_loss_;
          LOG(INFO) << "[DSEARCH] current_target_length_: " << this->current_target_length_;
          for (int param_id = 0; param_id < net_params.size(); ++param_id) {
            DynamicPrecisionParamSetLength(param_id, 0, inc_step);
          }
          for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
            DynamicPrecisionRoundSetLength(round_layer_id, 0, inc_step);
          }
        }
        this->num_loss_increasing_ = 0;
      } else {
        this->num_loss_increasing_++;
      }
    } else {
      this->num_loss_increasing_ = this->param_.dsearch_window();
      if (this->smoothed_losses_[idx] < this->min_smoothed_loss_) {
        int ilength = this->net_->params_ilength()[0];
        int flength = this->net_->params_flength()[0];
        if (ilength+flength == this->param_.max_length()) {
          LOG(INFO) << "Iteration " << this->iter_
                    << ", [DSEARCH] loss is reduced below the last min value";
          LOG(INFO) << "[DSEARCH] bit length will be reduced to: " << this->current_target_length_;
          int inc_step = this->current_target_length_ - this->param_.max_length();
          for (int param_id = 0; param_id < net_params.size(); ++param_id) {
            DynamicPrecisionParamSetLength(param_id, 0, inc_step);
          }
          for (int round_layer_id = 0; round_layer_id < net_round_layers.size(); ++round_layer_id) {
            DynamicPrecisionRoundSetLength(round_layer_id, 0, inc_step);
          }
        }
      }
    }
    break;
  }
  default:
    LOG(FATAL) << "Unknown solver dynamic precision stage: " << this->dynamic_precision_stage_;
  }
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::DynamicPrecisionParamSetLength(int param_id, int ilength_diff, int flength_diff) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  int layer_id = this->net_->learnable_param_layer_ids()[param_id];
  int flength = this->net_->params_flength()[param_id];
  int ilength = this->net_->params_ilength()[param_id];
  int new_flength = flength+flength_diff;
  int new_ilength = ilength+ilength_diff;
  LOG(INFO) << "Iteration " << this->iter_
            << ", " << this->net_->layer_names()[layer_id] << ", "
            << net_params[param_id]->shape_string()
            << " ilength is changed: " << ilength << " --> " << new_ilength
            << ", flength is changed: " << flength << " --> " << new_flength;

  this->net_->set_params_flength(param_id, new_flength);
  this->net_->set_params_ilength(param_id, new_ilength);
  this->net_->set_params_epsilon(param_id);
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::DynamicPrecisionParamOverflowCheck(int param_id) {
  int layer_id = this->net_->learnable_param_layer_ids()[param_id];
  int under = this->net_->params_under()[param_id];
  int over = this->net_->params_over()[param_id];
  int diff_under = this->net_->params_diff_under()[param_id];
  int diff_over = this->net_->params_diff_over()[param_id];

  if(under > 0 || diff_under > 0 || over > 0 || diff_over > 0) {
    DynamicPrecisionParamSetLength(param_id, 1, -1);
    if (this->net_->params_dynamic_precision_stage()[param_id] == SolverDynamicPrecision::ISEARCH) {
      this->net_->set_params_dynamic_precision_stage(param_id, SolverDynamicPrecision::DSEARCH);
      LOG(INFO) << "Iteration " << this->iter_
                << ", [ISEARCH --> DSEARCH] for "
                << this->net_->layer_names()[layer_id];
    }
  }
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::DynamicPrecisionRoundOverflowCheck(int round_layer_id) {
  const vector<RoundLayer<Dtype>*>& net_round_layers = this->net_->round_layers();
  int layer_id = this->net_->round_layer_ids()[round_layer_id];
  int under =      net_round_layers[round_layer_id]->data_under();
  int over =       net_round_layers[round_layer_id]->data_over();
  int diff_under = net_round_layers[round_layer_id]->diff_under();
  int diff_over =  net_round_layers[round_layer_id]->diff_over();

  if(under > 0 || diff_under > 0 || over > 0 || diff_over > 0) {
    DynamicPrecisionRoundSetLength(round_layer_id, 1, -1);
    if (this->net_->round_layers_dynamic_precision_stage()[round_layer_id] == SolverDynamicPrecision::ISEARCH) {
      this->net_->set_round_layers_dynamic_precision_stage(round_layer_id, SolverDynamicPrecision::DSEARCH);
      LOG(INFO) << "Iteration " << this->iter_
                << ", [ISEARCH --> DSEARCH] for "
                << this->net_->layer_names()[layer_id];
    }
  }
}
// tna6 added
template <typename Dtype>
int SGDSolver<Dtype>::DynamicPrecisionParamIsearch(int param_id) {
  int under = this->net_->params_under()[param_id];
  int over = this->net_->params_over()[param_id];
  int diff_under = this->net_->params_diff_under()[param_id];
  int diff_over = this->net_->params_diff_over()[param_id];

  int isfsearch = 1;
  if(under > 0 || diff_under > 0 || over > 0 || diff_over > 0) {
  } else {
    if (this->net_->params_dynamic_precision_stage()[param_id] == SolverDynamicPrecision::ISEARCH) {
      int num_no_over = this->net_->params_num_no_over()[param_id];
      if(this->param_.isearch_window() == num_no_over) {
        DynamicPrecisionParamSetLength(param_id, -1, 1);
        this->net_->set_params_num_no_over(param_id, 0);
      } else {
        this->net_->set_params_num_no_over(param_id, ++num_no_over);
      }
      isfsearch = 0;
    }
  }
  return isfsearch;
}

// tna6 added
template <typename Dtype>
void SGDSolver<Dtype>::DynamicPrecisionRoundSetLength(int round_layer_id, int ilength_diff, int flength_diff) {
  const vector<RoundLayer<Dtype>*>& net_round_layers = this->net_->round_layers();
  int layer_id = this->net_->round_layer_ids()[round_layer_id];
  const vector<Blob<Dtype>*>& round_top = this->net_->top_vecs()[layer_id];
  int flength =    net_round_layers[round_layer_id]->flength();
  int ilength =    net_round_layers[round_layer_id]->ilength();
  int new_flength = flength+flength_diff;
  int new_ilength = ilength+ilength_diff;
  LOG(INFO) << "Iteration " << this->iter_
            << ", " << net_round_layers[round_layer_id]->layer_param().name()
            << " Top data, " << round_top[0]->shape_string()
            << " ilength is changed: " << ilength << " --> " << new_ilength
            << ", flength is changed: " << flength << " --> " << new_flength;

  net_round_layers[round_layer_id]->set_flength(new_flength);
  net_round_layers[round_layer_id]->set_ilength(new_ilength);
  net_round_layers[round_layer_id]->set_epsilon();
}

// tna6 added
template <typename Dtype>
int SGDSolver<Dtype>::DynamicPrecisionRoundIsearch(int round_layer_id) {
  const vector<RoundLayer<Dtype>*>& net_round_layers = this->net_->round_layers();
  int under =      net_round_layers[round_layer_id]->data_under();
  int over =       net_round_layers[round_layer_id]->data_over();
  int diff_under = net_round_layers[round_layer_id]->diff_under();
  int diff_over =  net_round_layers[round_layer_id]->diff_over();

  int isfsearch = 1;
  if(under > 0 || diff_under > 0 || over > 0 || diff_over > 0) {
  } else {
    if (this->net_->round_layers_dynamic_precision_stage()[round_layer_id] == SolverDynamicPrecision::ISEARCH) {
      int num_no_over = this->net_->round_layers_num_no_over()[round_layer_id];
      if(this->param_.isearch_window() == num_no_over) {
        DynamicPrecisionRoundSetLength(round_layer_id, -1, 1);
        this->net_->set_round_layers_num_no_over(round_layer_id, 0);
      } else {
        this->net_->set_round_layers_num_no_over(round_layer_id, ++num_no_over);
      }
      isfsearch = 0;
    }
  }
  return isfsearch;
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
