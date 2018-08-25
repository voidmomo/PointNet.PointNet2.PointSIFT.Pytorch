# coding:utf-8

import os
import glob
import torch
from torch.utils.ffi import create_extension
import shutil

base_path = os.path.dirname(os.path.abspath(__file__))


def build():
    headers = [h for h in glob.glob('cinclude/pointsift.h')]
    sources = [s for s in glob.glob('csrc/pintsift.c')]
    defines = []
    with_cuda = False

    if torch.cuda.is_available():
        sources += [s for s in glob.glob(('csrc/pointsift.cu'))]
        defines += [('WITH_CUDA', None)]
        with_cuda = True

    ffi = create_extension(
        name='_ext.pointnet_extension',
        headers=headers,
        sources=sources,
        relative_to=__file__,
        with_cuda=with_cuda,
        define_macros=defines,
        extra_compile_args=["-std=c99"]
    )
    #
    # ffi = create_extension(
    #     '_ext.pointnet2',
    #     headers=[a for a in glob.glob("cinclude/*_wrapper.h")],
    #     sources=[a for a in glob.glob("csrc/*.c")],
    #     define_macros=[('WITH_CUDA', None)],
    #     relative_to=__file__,
    #     with_cuda=False,
    #     #extra_objects=extra_objects,
    #     include_dirs=[os.path.join(base_path, 'cinclude')],
    #     verbose=False,
    #     package=False
    # )
    ffi.build()


def clean():
    shutil.rmtree(os.path.join(base_path, '_ext'))


if __name__ == '__main__':
    build()
