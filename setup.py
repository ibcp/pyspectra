from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyspectra',
    version='0.0.0',
    description='A python package for working with spectral data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
       'Development Status :: 1 - Planning',

       'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

       'Programming Language :: Python :: 3 :: Only',
       'Programming Language :: Python :: 3.6',
       'Programming Language :: Python :: 3.7',

       'Operating System :: OS Independent',

       'Intended Audience :: Science/Research',
       'Topic :: Scientific/Engineering',
    ],
    keywords='spectroscopy Raman SERS FTIR vibrational processing analysis',
    url='https://github.com/rguliev/pyspectra',
    author='Rustam Guliev',
    author_email='glvrst@gmail.com',
    license='GPLv3+',
    packages=setuptools.find_packages(),
    #packages=['pyspectra'],
    python_requires='~=3.6',
    install_requires=[
      'numpy',
      'pandas'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    )