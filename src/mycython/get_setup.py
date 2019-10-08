####!/usr/local/bin/python3
import sys
import os
import numpy as np
import cython_gsl
import newflow.lib.myoptik as myoptik

_O = myoptik.setup_opt()
executer = False
module = _O.module
sources = ""
rmfiles = ""
for cf in _O.pfiles:
    sources += f'"{cf}"'+','
    rmfiles += f'{cf[:-4]}.c '
if len(_O.cfiles) != 0:
    for cf in _O.cfiles:
        sources += f'"{cf}"'+','
sources = sources[:-1]

# if len(sys.argv)>=3 and sys.argv[2][-2:]==".c":
#     sources = f'"{sys.argv[1]}","{sys.argv[2]}"'
# else:
#     sources = f'"{sys.argv[1]}"'

#from sysconfig import get_paths
#from pprint import pprint

#info = get_paths()  # a diction
#input(info['include'])

numpy_I = np.get_include()
gls_I = cython_gsl.get_include()
lib_dir = os.getcwd()
lib_gls = cython_gsl.get_library_dir()
cython_include = cython_gsl.get_cython_include_dir()
setup = f"""
import os
import sys
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import cython_gsl

##Whether or not to include docstring in the Python extension. If False, the binary size will be smaller
##but the __doc__ attribute of any class or function will be an empty string.

Options.docstrings = True

## generer un fichier html
Options.annotate = True

## arreter a la premiere erreur de compilation
Options.fast_fail = True

## verification que des arguments n'ont pas la valeur None....
# Options.nonecheck=True
ext = Extension(
        name="{module}",
        sources=[{sources}],
        include_dirs = [".", "{numpy_I}","{gls_I}"],
        library_dirs = ["{lib_dir}","{lib_gls}"],
        libraries=cython_gsl.get_libraries(),
        cython_include_dirs=[cython_gsl.get_cython_include_dir()]
	)
setup(ext_modules=cythonize(ext))

#os.system(f"mv {module}.cpython-37m-darwin.so {module}.so")
#os.system(f"rm -r build {rmfiles}")

"""
import posixpath
yes = True
if posixpath.isfile("setup.py"):
    if(input("voullez-vous ecraser le fichier setup.py?[n]") == "n"):
        yes = False
if yes:
    with open("setup.py", "w") as f:
        f.write(setup)
if _O.build:
    import os
    os.system("python setup.py build_ext -i")
else:
    print("executer:\n python setup.py build_ext -i")
