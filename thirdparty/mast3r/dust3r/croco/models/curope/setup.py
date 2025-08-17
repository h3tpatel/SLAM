# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def get_extension():
    """Build a CUDA or C++ extension depending on CUDA availability."""
    if cuda.is_available():
        all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
        return CUDAExtension(
            name="curope",
            sources=["curope.cpp", "kernels.cu"],
            extra_compile_args={
                "nvcc": ["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                "cxx": ["-O3"],
            },
        )

    # CPU-only build: the C++ file already contains a CPU implementation.
    return CppExtension(name="curope", sources=["curope.cpp"], extra_compile_args=["-O3"])


setup(
    name="curope",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
)

