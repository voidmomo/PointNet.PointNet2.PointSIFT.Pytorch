# coding:utf-8

import os
import glob
import torch
from torch.utils.ffi import create_extension
import shutil

base_path = os.path.dirname(os.path.abspath(__file__))


def build():
    headers = [h for h in glob.glob('cinclude/*_wrapper.h')]
    sources = [s for s in glob.glob('csrc/*.c')]
    if torch.cuda.is_available():
        sources += [s for s in glob.glob(('csrc/*.cu'))]
    ffi = create_extension(
        name='_ext.pointnet_extension',
        headers=headers,
        sources=sources,
        relative_to=__file__,
        with_cuda=True,
        include_dirs=[os.path.join(base_path, 'cinlcude')],
        verbose=True,
        package=False
    )
    ffi.bulid()


def clean():
    shutil.rmtree(os.path.join(base_path, '_ext'))


if __name__ == '__main__':
    build()
