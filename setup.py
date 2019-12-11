from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

__version__ = '0.0.1'

CTLDIR = 'ctl/modules/src/'

ext_modules = [
    Extension(
        'ctl',
        [CTLDIR+'img/projectiondata.cpp',
         CTLDIR+'img/singleviewdata.cpp',
         CTLDIR+'mat/homography.cpp',
         CTLDIR+'mat/matrix_algorithm.cpp',
         CTLDIR+'mat/projectionmatrix.cpp',
         CTLDIR+'ocl/openclconfig.cpp',
         CTLDIR+'ocl/cldirfileloader.cpp',
         CTLDIR+'processing/radontransform2d.cpp',
         'src/'+'radon2d_kernel.cpp',
         'src/'+'parallelsetup.cpp',
         'src/'+'simple_backprojector_kernel.cpp',
         'src/'+'simplebackprojector.cpp',
         'src/'+'pybind_radon.cpp'],
        include_dirs=[
            'include',
            CTLDIR,
        ] + cpp_extension.include_paths(cuda=True),
        language='c++',
        libraries=['OpenCL', 'c10', 'caffe2', 'torch', 'torch_python', '_C'],
        library_dirs=cpp_extension.library_paths(cuda=True),
        define_macros=[('OCL_CONFIG_MODULE_AVAILABLE', None), ('NOQT', None)]
    ),
]

setup(
    name='ctl',
    version=__version__,
    author='Philipp Ernst',
    author_email='phil23940@yahoo.de',
    description='2D Radon transform using CTL',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)