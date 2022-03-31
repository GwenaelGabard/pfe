import os
import subprocess
from setuptools import setup, Command


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

class CreateTables(Command):
    description = 'Generate look-up tables'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass
            
    def run(self):
        print("Generating quadrature rules...")
        subprocess.call("python make_quadrature.py", shell=True, cwd="pfe/quadrature")
        print("Generating look-up tables for basis functions...")
        subprocess.call("python make_basis.py", shell=True, cwd="pfe/shape")

setup(
    name = "pfe",
    version = "0.1.0",
    author = "Gwénaël Gabard",
    author_email = "gwenael.gabard@univ-lemans.fr",
    description = ("Python Finite Elements"),
    license = "MIT",
    keywords = "finite element, numerical simulation, computational modelling",
    url = "https://github.com/GwenaelGabard/pfe",
    packages=['pfe'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    cmdclass={'create_tables': CreateTables,},
)
