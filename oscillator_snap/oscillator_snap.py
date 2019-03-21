import os
import dill
import numpy as np
import copy
from math import pi, log, exp, sqrt, floor
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras import optimizers

from oscillator_snap.oscillator_auxiliaries import *


def compile_model(model, learning_rate):
	"""compiles the model
	
	:param model: RNN model
	:returns: compiled model"""
	# optimizer (stochastic gradient descent)
	sgd = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
	# compile model
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
	return model


def generate_model(data_dim_in, data_dim_out, past, nodes, learning_rate, cell=LSTM, n_hidden_layers=1):
	"""generates the model with n hidden layers (does not compile it)
	
	:param n_hidden_layers: number of hidden layers
	:returns: return the keras model"""
	model = Sequential()
	if(n_hidden_layers > 0): 
		model.add(cell(nodes, activation='tanh', return_sequences=True, input_shape=(past, data_dim_in)))  # input layer
		for i in range(n_hidden_layers-1):
			model.add(cell(nodes, activation='tanh', return_sequences=True))  # hidden layers
		model.add(cell(nodes, activation='tanh'))  # last hidden layer
	else:
		model.add(cell(nodes, activation='tanh', input_shape=(past, data_dim_in)))  # input layer
	model.add(Dense(data_dim_out, activation='linear'))  # output layer
	return model


def parse_train_data(seq, past, dim_in, dim_out):
	"""parses the training data for the RNN
	
	:param seq: the sequence to be parsed
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param dim_out: the dimensionality of the prediction
	:returns: input data X and prediction Y""" 
	# network input output
	X = []
	Y = []
	# determine the length of the sequence
	L = len(seq[0])
	# take care of dimensions
	seq_in = seq[0:dim_in]
	seq_out = seq[0:dim_out]
	# reshape
	seq_in = np.array(seq_in)
	seq_in = seq_in.transpose()
	seq_out = np.array(seq_out)
	seq_out = seq_out.transpose()
	# organize
	for i in range(L-past):
		X.append(seq_in[i:i+past])
		Y.append(seq_out[i+past])
	return np.array(X), np.array(Y)


def forecast_starter(seq, past, dim_in):
	"""prepares the staring vector for RNN forecast
	
	:param seq: the sequence from which the beggining is used as the forcast starter
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:returns: starter input data""" 
	# determine the length of the sequence
	L = len(seq[0])
	# take care of dimensions
	seq_in = seq[0:dim_in]
	# reshape
	seq_in = np.array(seq_in)
	seq_in = seq_in.transpose()
	return np.array(seq_in[0:past])


def update_x(model, x, new_input, number_of_variables=1):
	"""update x and make the next prediction y (auxiliary for forecast) 
	
	:param model: RNN model
	:param x: input vector to RNN
	:param new_input: value of the new input
	:param number_of_variables: the number of variables that have to be updated (default 1)
	:returns: updated x""" 
	y = (model.predict(x))[0][0:number_of_variables] # next prediction value
	# update x
	x = x[0][1:]
	x = np.append(x, np.append(y, new_input)).reshape(1, x.shape[0]+1, x.shape[1])
	return x, y


def forecast(model, past, dim_in, stream_start, time_span, oscillator_input, number_of_variables=1):
	"""forecasts the signal, using the RNN
	
	:param model: RNN model
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param forcast_start: the initial input to the network
	:param time_span: the time span of the forcasting, in network steps
	:param oscillator_input: the time series of the input being fed to the oscillator while forcasting (starting from the time of the first forcasted value)
	:param number_of_variables: the number of variables that are being forecasted (default 1)
	:returns: forcast of the time series""" 
	print("forecast")
	s_for = []
	x = np.array([stream_start])
	for i in range(time_span):
		x, y = update_x(model, x, oscillator_input[i], number_of_variables=number_of_variables)
		s_for.append(y.tolist()) # save the values
	return s_for


def period_measure(model, past, dim_in, forcast_start, constant_input_offset, thr=0.0, period_counts=100):
	"""estimate the natural period from the RNN model, also useful for chaotic and quasi-periodic dynamics because of averaging (in units of dt*sampling)
	
	:param model: RNN model
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param forcast_start: the starting point of the time series
	:param constant_input_offset: the constant input to be fed to the network
	:param thr: signal threshold (default 0.0)
	:param period_counts: how many periods do we average over (default 100)
	:returns: the natural period""" 
	print("period estimation")
	x = np.array([forcast_start])
	yh, y = 0, 0 # initial setting of signal and its previous value
	# first warmup
	for t in range(1000):
		yh = y # previous value
		x, y = update_x(model, x, constant_input_offset) # update x,y
	# then we get to a position just after a crossing
	failsafe_time = 0
	while((yh < thr and y > thr) == False):
		yh = y # previous value
		x, y = update_x(model, x, constant_input_offset) # update x,y
		# time update
		failsafe_time = failsafe_time + 1
		# if it runs for too long it might be that it does not oscillate
		if(failsafe_time == 1000): # the choice 1000 is arbitrary
			print("\tallert: it will probably never reach the threshold")
			return
	previous_crossing = (thr-yh)/(y-yh)*1 # time of crossing correction
	# now set time to zero and look for following crossings
	time = 0
	avg_period = 0
	for p in range(period_counts):
		yh = thr+1 # break the condition
		failsafe_time = 0
		while((yh < thr and y > thr) == False):
			yh = y # previous value
			x, y = update_x(model, x, constant_input_offset) # update x,y
			# time update
			time = time + 1
			failsafe_time = failsafe_time + 1
			# if it runs for too long it might be that it does not oscillate
			if(failsafe_time == 1000): # the choice 1000 is arbitrary
				print("\tallert: it will probably never reach the threshold")
				return
		crossing = time + (thr-yh)/(y-yh)*1 # time of crossing
		avg_period = avg_period + crossing-previous_crossing # add to the period
		previous_crossing = crossing # reset previous crossing
	avg_period = avg_period/period_counts
	return avg_period


def PRC_measure(model, past, dim_in, forcast_start, constant_input_offset=0.0, warmup_time=1000, warmup_between_probes=100, stimulation=0.25, thr=0.0, period_counts=5, phase_repeats=20):
	"""estimate the PRC from the RNN model (in units of sampling)
	
	:param model: RNN model
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param forcast_start: the starting point of the time series
	:param constant_input_offset: the constant input to be fed to the network (default 0.0)
	:param warmup_time: the time for the system to relax to its steady state (default 1000)
	:param warmup_between_probes: the time for the system to relax between diferent phase probes (default 100)
	:param stimulation: how strong to stimulate the oscillator (default 0.05)
	:param thr: signal threshold (default 0.0)
	:param period_counts: how many periods do we average over (default 5)
	:param phase_repeats: how many times to stimulate at each phase (default 10)
	:returns: the PRC in list format""" 
	print("PRC estimation")
	period = period_measure(model, past, dim_in, forcast_start, constant_input_offset, thr=thr)
	PRC = [[2*pi*(i+1)/period for i in range(floor(period)-1)],[0 for i in range(floor(period)-1)]] # PRC list (i+1 because when i=0 the hit comes effectively between 0.5 and 1.5 depending on how it crosses the threshold)
	x = np.array([forcast_start])
	yh, y = 0, 0 # initial setting of signal and its previous value
	# first warmup
	for t in range(warmup_time):
		yh = y # previous value
		x, y = update_x(model, x, constant_input_offset) # update x,y
	for ph in range(floor(period)-1):
		print("\tphase = ", ph, "/", floor(period)-1-1)
		# warmup between different phase probes
		for t in range(warmup_between_probes):
			yh = y # previous value
			x, y = update_x(model, x, constant_input_offset) # update x,y
		# set phase shift to 0, then we will add to it to get an average
		phase_shift = 0
		for r in range(phase_repeats):
			# then we get to a position just after a crossing
			while((yh < thr and y > thr) == False):
				yh = y # previous value
				x, y = update_x(model, x, constant_input_offset) # update x,y
			first_crossing = (thr-yh)/(y-yh)*1 # time of crossing correction
			# now set time to zero and look for following crossings
			time = 0
			# wait ph steps...
			for t in range(ph):
				yh = y # previous value
				x, y = update_x(model, x, constant_input_offset) # update x,y
				# time update
				time = time + 1
			# and then stimulate
			yh = y # previous value
			x, y = update_x(model, x, constant_input_offset+stimulation) # notice +stimulation
			# time update
			time = time + 1
			# now run for some periods and then evaluate the phase shift
			for p in range(period_counts):
				yh = thr+1 # break the condition (just in case - generally should be already broken)
				while((yh < thr and y > thr) == False):
					yh = y # previous value
					x, y = update_x(model, x, constant_input_offset) # update x,y
					# time update
					time = time + 1
				crossing = time + (thr-yh)/(y-yh)*1 # time of crossing
			# altered periods
			altered_periods = crossing-first_crossing
			# phase shift
			phase_shift = phase_shift + 2*pi*(altered_periods-period_counts*period)/period
		phase_shift = phase_shift/phase_repeats
		PRC[1][ph] = phase_shift/stimulation
	return PRC


def lyapunov_measure(model, past, dim_in, forcast_start, constant_input_offset=0.0, warmup_time=1000, delta=0.005, tau=10, unconsidered_trials=50, trials=500):
	"""estimate the largest Lyapunov exponent from the RNN model (in units of 1/(sampling*dt) ), 
	this is achieved by staring with two close trajectories (reference one - x, and perturbed one x_p), evolving them and evaluating the deviation. 
	To assure we are measuring the maximal exponent the perturbed trajectory is every time renormalized 
	(the whole past is rescaled by delta*sqrt(1/past*sum_i^past (x_p(t-i)-x(t-i))^2) ), 
	thus allowing the maximal exponent to take over and there's no need for any embedding - pretty neat:)
	
	:param model: RNN model
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param forcast_start: the starting point of the time series
	:param constant_input_offset: the constant input to be fed to the network (default 0.0)
	:param warmup_time: the time for the system to relax to its steady state (default 1000)
	:param delta: the starting deviation (default 0.005)
	:param tau: the nuber of updates before the deviation is evaluated (default 10)
	:param uncosidered_trials: the number of trials that are not considered in the averaging, alowing the maximal Lyapunov exponent to take over (default 50)
	:param trials: the number of trials considered in the averaging (default 500)"""
	print("Lyapunov estimation")
	# warmup 
	x = np.array([forcast_start])
	for t in range(warmup_time):
		x, y = update_x(model, x, constant_input_offset) # update x,y
	# copy the trajectory
	x2 = x.copy()
	# first step, x2 gets perturbed slightly
	x, y = update_x(model, x, constant_input_offset) # update x,y
	x2, y2 = update_x(model, x2, constant_input_offset+delta) # update x2,y2, notice +delta
	for utr in range(unconsidered_trials): # unconsidered trials
		for t in range(tau):
			x, y = update_x(model, x, constant_input_offset) # update x,y
			x2, y2 = update_x(model, x2, constant_input_offset) # update x2,y2
		# rescale
		rescaling = sqrt(1/past*sum((x2[0][i][0]-x[0][i][0])**2 for i in range(len(x[0]))))
		x2 = x + delta/rescaling*(x2-x)
	lyapunov = 0
	for tr in range(trials): # considered trials	
		for t in range(tau):
			x, y = update_x(model, x, constant_input_offset) # update x,y
			x2, y2 = update_x(model, x2, constant_input_offset) # update x2,y2
		# rescale
		rescaling = sqrt(1/past*sum((x2[0][i][0]-x[0][i][0])**2 for i in range(len(x[0]))))
		x2 = x + delta/rescaling*(x2-x)
		# averaging deviation
		lyapunov += log(rescaling/delta)
	return lyapunov/tau/trials


def bifurcation_diagram(model, past, dim_in, forcast_start, ci_min, ci_max, dci, time_window=500):	
	"""measure the bifurcation diagram from the RNN model
	
	:param model: RNN model
	:param past: the number of considered steps in the past
	:param dim_in: the dimensionality of the data fed to the network
	:param forcast_start: the starting point of the time series
	:param ci_min: minimal (starting) value of the constant input - bifurcation parameter
	:param ci_max: maximal (ending) value of the constant input - bifurcation parameter
	:param dci: bifurcatin parameter step (ci - constant input)
	:param time_window: the time window in which the diagram is evaluated for each bifurcation parameter value, in network steps (default 500)
	:returns: the scatter plot of the bifurcation diagram""" 
	print("bifurcation diagram")
	bif = [[],[]] # bifurcation list
	x = np.array([forcast_start])
	# first warmup
	for t in range(1000):
		x, y = update_x(model, x, ci_min) # update x,y
	# bp loop (bifurcation parameter)
	for bp in [ci_min+dci*i for i in range(floor((ci_max-ci_min)/dci))]:
		print("\tbp = ", "%.3f" % bp, "/", ci_max)
		yhhhh = y # pre-pre-pre-previous value
		yhhh = y # pre-pre-previous value
		yhh = y # pre-previous value
		yh = y # previous value
		# warmup
		for t in range(100):
			x, y = update_x(model, x, bp) # update x,y
		for t in range(time_window):
			yhhhh = yhhh
			yhhh = yhh
			yhh = yh
			yh = y
			x, y = update_x(model, x, bp) # update x,y
			if(yhhhh < yhhh and yhhh < yhh and yhh > yh and yh > y): # 5 point local maximum
				bif[0].append(bp)
				bif[1].append(yhh)	
	return bif

	
def save_model_h5py(model, save_path="model_save"):
	"""saves a keras model to h5py file (also makes the 'model_save/' directory if it wasn't there before)
	
	:param model: model to save
	:param save_path: path where to save model, staring from the directory of the simulation (default "model_save")"""
	# directory
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# serialize model to JSON
	model_json = model.to_json()
	with open(save_path+"/model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(save_path+"/model.h5")


def save_model_dill(model, save_path="model_save"):
	"""saves a keras model with dill (also makes the 'model_save/' directory if it wasn't there before)
	
	:param model: model to save
	:param save_path: path where to save model, staring from the directory of the simulation (default "model_save")"""
	# directory
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	config = model.get_config()
	weights = model.get_weights()
	# pickling
	fc = open(save_path+'/config.pickle', 'wb')
	fw = open(save_path+'/weights.pickle', 'wb')
	dill.dump(config, fc)
	dill.dump(weights, fw)
	fc.close()
	fw.close()


def load_model_h5py(load_path="model_save"):
	"""loads model with h5py file (from 'model_save/' directory)
	
	:param load_path: path where to load model from, staring from the directory of the simulation (default "model_save")
	:returns: keras model"""
	# load json and create model
	json_file = open(load_path+'/model.json', 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights(load_path+"/model.h5")
	return model


def load_model_dill(load_path="model_save"):
	"""loads model with dill (from 'model_save/' directory)
	
	:param load_path: path where to load model from, staring from the directory of the simulation (default "model_save")
	:returns: keras model"""
	# unpickling
	fc = open(load_path+'/config.pickle', 'rb')
	fw = open(load_path+'/weights.pickle','rb')
	config = dill.load(fc)
	weights = dill.load(fw)
	fc.close()
	fw.close()
	# make model
	model = Sequential.from_config(config)
	model.set_weights(weights)
	return model
	
