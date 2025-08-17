#include "gn.h"

std::vector<torch::Tensor> refine_matches_cpu(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    const int radius,
    const int dilation_max) {
  TORCH_CHECK(false, "refine_matches_cpu not implemented");
  return {};
}

std::vector<torch::Tensor> iter_proj_cpu(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh) {
  TORCH_CHECK(false, "iter_proj_cpu not implemented");
  return {};
}
