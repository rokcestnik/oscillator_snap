from setuptools import setup

setup(
	name='oscillator_snap',
	packages=['oscillator_snap'],
	version='1.0',
	description='Make a snapshot of an oscillator through its time series',
	long_description=open('README.rst').read(),
	author='Rok Cestnik',
	author_email='rokcestn@uni-potsdam.de',
	license='MIT',
	install_requires=[
		'tensorflow',
		'keras',
		'numpy',
		'scipy',
		'h5py',
		'dill'
	],
)
