#include "gn.h"

std::vector<torch::Tensor> gauss_newton_points_cpu(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj,
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {
  TORCH_CHECK(false, "gauss_newton_points_cpu not implemented");
  return {};
}

std::vector<torch::Tensor> gauss_newton_rays_cpu(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj,
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {
  TORCH_CHECK(false, "gauss_newton_rays_cpu not implemented");
  return {};
}

std::vector<torch::Tensor> gauss_newton_calib_cpu(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj,
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {
  TORCH_CHECK(false, "gauss_newton_calib_cpu not implemented");
  return {};
}
