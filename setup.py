from setuptools import setup, Extension
import pybind11
from glob import glob

ext_modules = [
    Extension(
        "cpp_package",  # this is the Python import name: `import cpp_package`
        sources=[
            "cpp_package/src/bindings.cpp",
            *glob("cpp_package/src/*.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            "cpp_package/include",
        ],
        language="c++",
    )
]

setup(
    name="cpp_package",
    version="0.1",
    ext_modules=ext_modules,
)
