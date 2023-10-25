#include <torch/extension.h>


__global__ void d_mult(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * (*b);
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int block_size = 128;


__forceinline__ int calc_grid_size(int m) {
    return (m + block_size - 1) / block_size;
}


torch::Tensor mult(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);

    auto c = torch::empty_like(a);
    int n = a.numel();

    d_mult<<<calc_grid_size(n), block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prod", &mult, "Custom vector multiply");
}
