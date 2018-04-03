from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
nn_plugin_header_dir = '@NNPLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NNPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_openmmnn',
                      sources=['NNPluginWrapper.cpp'],
                      libraries=['OpenMM', 'NNPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmnn',
      version='1.0',
      py_modules=['openmmnn'],
      ext_modules=[extension],
     )
