# -*- coding: utf-8 -*-

import io
import os
import os.path as osp

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package meta-data.
NAME = "tools_xue"
AUTHOR = 'xue'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
about = {}
about['__version__'] = VERSION

src_files = ['src/bvh.cpp', 'src/bvh_cuda_op.cu', "src/sampling.cpp", "src/dbscan.cpp", "src/dbscan_gpu.cu", "src/sampling_gpu.cu", "src/bindings.cpp"]
include_dirs = torch.utils.cpp_extension.include_paths() + [
    osp.join(here, 'include'),
    osp.expandvars('cuda-samples/Common')]
print(include_dirs)

extra_compile_args = {'nvcc': ['-DPRINT_TIMINGS=0',
                                   '-DDEBUG_PRINT=0',
                                   '-DERROR_CHECKING=1',
                               #'-DNUM_THREADS=256',
                                   '-DBVH_PROFILING=0',
                                   "-Xfatbin",
                                   "-compress-all"
                                   ],
                          'cxx': []}
extension = CUDAExtension('tools_xue_cuda',
                              src_files,
                              include_dirs=include_dirs,
                              extra_compile_args=extra_compile_args)

setup(name=NAME,
      version=about['__version__'],
      author=AUTHOR,
      python_requires=REQUIRES_PYTHON,
      packages=find_packages(),
      ext_modules=[extension],
      install_requires=[
          'torch>=1.4',
      ],
      cmdclass={'build_ext': BuildExtension})
