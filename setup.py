"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from distutils.util import convert_path
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

main_ns = {}
ver_path = convert_path('cogrecon/_version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


def read(fname):
    return open(path.join(path.dirname(__file__), fname)).read()


# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='msl-iposition-pipeline',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    # Note: Change version in cogrecon __init__.py
    version=main_ns['__version__'],

    description='A package for analyzing reconstruction data generated from iPosition and related memory tasks.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/kevroy314/msl-iposition-pipeline',

    # Author details
    author='Kevin Horecka',
    author_email='kevin.horecka@gmail.com',
    maintainer='Kevin Horecka',
    # License details
    license=read(path.abspath('LICENSE')),
    include_package_data=True,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Researchers',
        'Topic :: Scientific/Engineering :: Information Analysis',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[x.strip() for x in read('requirements.txt').split('\n')]
)
