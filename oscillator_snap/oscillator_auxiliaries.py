import os
import dill
import numpy as np
from math import pi, exp, atan, sqrt, acos


def angle(x, y):
	"""calculates the angle of (x,y) with respect to (0,0)
	
	:param x: x choordinate
	:param y: y choordinate
	:returns: the angle"""
	at = atan(y/x)
	if(x < 0): return at+pi
	elif(y < 0): return at+2*pi
	return at


def angle_between_vectors(v, w):
	"""calculates the angle between two vectors (any dimensional)
	
	:param v: vector 1
	:param w: vector 2
	:returns: the angle"""
	scalar = sum(v[i]*w[i] for i in range(len(v)))
	if(scalar/(L2_norm(v)*L2_norm(w)) > 1): return 0
	if(scalar/(L2_norm(v)*L2_norm(w)) < -1): return pi
	return acos(scalar/(L2_norm(v)*L2_norm(w)))


def L1_norm(seq):
	"""calculates the L1 norm of a sequence or vector
	
	:param seq: the sequence or vector
	:returns: the L1 norm"""
	norm = 0
	for i in range(len(seq)):
		norm += abs(seq[i])
	return norm


def L2_norm(seq):
	"""calculates the L2 norm of a sequence or vector
	
	:param seq: the sequence or vector
	:returns: the L2 norm"""
	norm = 0
	for i in range(len(seq)):
		norm += seq[i]**2
	return sqrt(norm)


def Linf_norm(seq):
	"""calculates the Linf norm of a sequence or vector
	
	:param seq: the sequence or vector
	:returns: the Linf norm"""
	largest = 0
	for i in range(len(seq)):
		if(abs(seq[i]) > largest): largest = abs(seq[i])
	return largest


def linear_interpolation(listx, listy, argument):
	"""calculates the linear interpolation of [listx,listy] at argument
	
	:param listx: x choordinates (should be ordered in ascending order)
	:param listy: y choordinates
	:param argument: where to evaluate the linear interpolation
	:returns: value of the linear interpolation at argument"""
	if(argument in listx):
		return listy[listx.index(argument)]
	if(argument < listx[0]):
		return listy[0] + (listy[0]-listy[1])/(listx[1]-listx[0])*(listx[0]-argument)
	if(argument > listx[-1]):
		return listy[-1] + (listy[-1]-listy[-2])/(listx[-1]-listx[-2])*(argument-listx[-1])
	index = 0
	while((listx[index] < argument and listx[index+1] > argument) != 1): index += 1
	return listy[index] + (listy[index+1]-listy[index])/(listx[index+1]-listx[index])*(argument-listx[index])


def gaussian(x, std=1):
	"""returns a Gaussian distribution
	
	:param x: variable
	:param std: standard deviation (default 1)
	:returns: Gaussian PDF"""
	return 1/(std*sqrt(2*pi))*exp(-x**2/(2*std**2))


def gaussian_der(x, std=1):
	"""returns a Gaussian derivative distrbution (x*PDF_normal)
	
	:param x: variable
	:param std: the standard deviation
	:returns: positive half of the Gaussian derivative PDF"""
	if(x < 0): return 0
	return (2-pi/2)/std**2*x*exp(-(2-pi/2)*x**2/(2*std**2))


def kernel_convolve(seq, kernel, kernel_width):
	"""convolves a sequence with a kernel
	
	:param seq: sequence to be convolved
	:param kernel: the kernel, a probability distribuion fucntion (does not have to be normalized)
	:param kernel_width: the width of the kernel (std for Gaussian)
	:returns: convolved sequence"""
	ker = np.array([kernel(i/kernel_width) for i in range(-round(3*kernel_width), round(3*kernel_width))])
	ker = ker/sum(ker)
	return np.convolve(seq, ker ,mode='same')


def save_object_dill(obj, filename, save_path="objects_save"):
	"""saves an object with dill (also makes the 'objects_save/' directory if it wasn't there)
	
	:param obj: object to save
	:param filename: filename without any extension
	:param save_path: path where to save object, staring from the directory of the simulation (default "objects_save")"""
	# directory
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# pickling
	f = open(save_path+"/"+filename+'.pickle', 'wb')	
	dill.dump(obj, f)
	f.close()
	
	
def load_object_dill(filename, load_path="objects_save"):
	"""loads an object with dill (from 'objects_save/' directory)
	
	:param filename: filename without any extension
	:param load_path: path where to load object from, staring from the directory of the simulation (default "objects_save")
	:returns: saved object"""
	# unpickling
	f = open(load_path+"/"+filename+'.pickle', 'rb')	
	obj = dill.load(f)
	f.close()
	return obj

