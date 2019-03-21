import numpy as np
from math import pi, log, exp, sqrt, floor, sin, cos, atan
import random

from oscillator_snap.oscillator_auxiliaries import *

	
def one_step_integrator(state, ders, I, dt):
	"""RK4 integrates state with derivative and input for one step of dt
	
	:param state: state of the variables
	:param ders: derivative functions
	:param I: input
	:param dt: time step
	:return: state after one integration step"""
	D = len(state)
	# 1
	k1 = [ders[i](state,I) for i in range(D)]
	# 2
	state2 = [state[i]+k1[i]*dt/2.0 for i in range(D)]
	k2 = [ders[i](state2,I) for i in range(D)]
	# 3
	state3 = [state[i]+k2[i]*dt/2.0 for i in range(D)] 
	k3 = [ders[i](state3,I) for i in range(D)]
	# 4
	state4 = [state[i]+k3[i]*dt for i in range(D)] 
	k4 = [ders[i](state4,I) for i in range(D)]
	# put togeather
	statef = [state[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0*dt for i in range(D)]
	return statef


def oscillator_period(ders, inp, warmup_time=1000.0, thr=0.0, dt=0.01):
	"""calculates the natural period of the oscillator
	
	:param ders: a list of state variable derivatives
	:param inp: the input to the system
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: natural period"""
	# initial conditions
	state = [0.0 for i in range(len(ders))]
	state[0] = 1.0
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# integration up to x = thr
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
	# Henon trick
	dt_beggining = 1.0/ders[0](state,inp)*(state[0]-thr)
	# spoil condition and go again to x = 0 (still counting time)
	xh = state[0]
	time = 0
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
		time = time + dt
	# another Henon trick
	dt_end = 1.0/ders[0](state,inp)*(state[0]-thr)
	return time + dt_beggining - dt_end


def oscillator_average_period(ders, inp, warmup_time=1000.0, avg_time=5000.0, thr=0.0, dt=0.01):
	"""calculates the average period of the oscillator (for chaotic or quasi-periodic)
	
	:param ders: a list of state variable derivatives
	:param inp: the input to the system
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:parma avg_time: averaging time (default 5000)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: average period"""
	# initial conditions
	state = [0.0 for i in range(len(ders))]
	state[0] = 1.0
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# integration up to x = thr
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
	# start counting periods and measuring time (taking into account the amount over x = 0)
	periods = 0
	time = (state[0]-thr)/ders[0](state,inp)
	# run for avg_time
	for i in range(round(avg_time/dt)):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
		time = time + dt
		if(state[0] > 0 and xh < 0): periods = periods+1
	# spoil condition and go again to x = 0 (still counting time)
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
		time = time + dt
	# correct for going over x = thr
	time = time - (state[0]-thr)/ders[0](state,inp)
	return time/(periods+1)


def oscillator_PRC(ders, inp=0.0, warmup_time=1000.0, stimulation=1.0, period_counts=5, dph=0.1, thr=0.0, dt=0.01):
	"""calculates the PRC
	
	:param ders: a list of state variable derivatives
	:param inp: the offset input to the system (default 0.0)
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param stimulation: strength of the stimulation (default 1.0)
	:param period_counts: how many periods to wait for evaluating the asymptotic phase shift (default 5)
	:param dph: phase resolution (default 0.1)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: average period"""
	period = oscillator_period(ders, inp)
	PRC = [[dph*i for i in range(floor(2*pi/dph))],[0 for i in range(floor(2*pi/dph))]] # PRC list
	# initial conditions
	state = [0.0 for i in range(len(ders))]
	state[0] = 1.0
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# stimulating phases
	for ph in [dph*i for i in range(floor(2*pi/dph))]:
		# integration up to x = thr
		xh = state[0]
		while((state[0] > thr and xh < thr) == False):
			xh = state[0]
			state = one_step_integrator(state, ders, inp, dt)
		# Henon trick
		dt_beggining = 1.0/ders[0](state,inp)*(state[0]-thr)
		# spoil condition and go to ph (counting time)
		xh = state[0]
		time = dt_beggining
		while(time < ph/(2*pi)*period):
			xh = state[0]
			state = one_step_integrator(state, ders, inp, dt)
			time = time + dt
		# stimulate 
		xh = state[0]
		state = one_step_integrator(state, ders, inp+stimulation, dt) # notice +stimulation
		time = time + dt
		#integrate for some periods
		for p in range(period_counts):
			xh = state[0] # spoil
			while((state[0] > thr and xh < thr) == False):
				xh = state[0]
				state = one_step_integrator(state, ders, inp, dt)
				time = time + dt
		# another Henon trick
		dt_end = 1.0/ders[0](state,inp)*(state[0]-thr)
		phase_shift = 2*pi*(time-dt_end - period_counts*period)/period
		PRC[1][round(ph/dph)] = phase_shift/stimulation
	return PRC


def oscillator_lyapunov(ders, inp=0.0, warmup_time=1000.0, delta=0.00001, tau=1.0, unconsidered_trials=100, trials=1000, dt=0.01):
	"""calculates the largest Lyapunov exponent
	
	:param ders: a list of state variable derivatives
	:param inp: the offset input to the system (default 0.0)
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param delta: the amplitude of the initial trajectory shift (default 0.00001)
	:param tau: the time for which the two trajectories are left to deviate (default 1.0)
	:param unconsidered_trials: the number of trials that are not to be taken into account when averaging but allow the largest component of the Lyapunov vector to become dominant (default 100)
	:param trials: the number of trials over which we average to get an estimation of the Lyapunov exponent (default 1000)
	:param dt: time step (default 0.01)
	:returns: largest Lyapunov exponent"""
	# initial conditions
	state = [0.0 for i in range(len(ders))]
	state[0] = 1.0
	# warmup
	for t in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# other trajectory init
	state2 = state[:]
	state2[0] += delta # deviation
	# uncosidered trials
	for i in range(unconsidered_trials):
		for t in range(round(tau/dt)):
			state = one_step_integrator(state, ders, inp, dt)
			state2 = one_step_integrator(state2, ders, inp, dt)
		diff = np.array(state)-np.array(state2) # difference
		diff = delta*diff/L2_norm(diff) # normalize (to delta)
		state2 = [state[0]+diff[0], state[1]+diff[1], state[2]+diff[2]]# readjust state2
	# cosidered trials
	lyapunov = 0.0
	for i in range(trials):
		for t in range(round(tau/dt)):
			state = one_step_integrator(state, ders, inp, dt)
			state2 = one_step_integrator(state2, ders, inp, dt)
		diff = np.array(state)-np.array(state2) # difference
		lyapunov += log(L2_norm(diff)/delta) # averaging!
		diff = delta*diff/L2_norm(diff) # normalize (to delta)
		state2 = [state[0]+diff[0], state[1]+diff[1], state[2]+diff[2]]# readjust state2
	return lyapunov/tau/trials


def oscillator_bifurcation(ders, inp_min, inp_max, d_inp, warmup_time=1000.0, short_warmup=200.0, time_window=500, dt=0.01):
	"""calculates the bifurcation diagram with respect to the input
	
	:param ders: a list of state variable derivatives
	:param inp_min: the min input
	:param inp_max: the max input
	:param d_inp: input resolution
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param short_warmup: the time for relaxing between different inputs (default 200)
	:param time_window: time window (default 500)
	:param dt: time step (default 0.01)
	:returns: bifurcation diagram list"""
	print("Calculating biffurcation diagram...")
	bif = [[],[]] # bifurcation list
	# initial conditions
	state = [0.0 for i in range(len(ders))]
	state[0] = 1.0
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp_min, dt)
	# inp loop
	for inp in [inp_min+d_inp*i for i in range(floor((inp_max-inp_min)/d_inp))]:
		print("\tb = ", "%.3f" % inp, "/", inp_max)
		xhhhh = state[0] # pre-pre-pre-previous value
		xhhh = state[0] # pre-pre-previous value
		xhh = state[0] # pre-previous value
		xh = state[0] # previous value
		# warmup
		for i in range(round(short_warmup/dt)):
			state = one_step_integrator(state, ders, inp, dt)
		for t in range(round(time_window/dt)):
			xhhhh = xhhh
			xhhh = xhh
			xhh = xh
			xh = state[0]
			state = one_step_integrator(state, ders, inp, dt)
			if(xhhhh < xhhh and xhhh < xhh and xhh > xh and xh > state[0]): # 5 point local maximum
				bif[0].append(inp)
				bif[1].append(xhh)
	return bif


def generate_signal(ders, n, sampling, number_of_variables=1, warmup_time=1000.0, tau=3.0, eps=0.5, dt=0.01):
	"""generates signal for the oscillator driven by correlated noise
	
	:param ders: a list of state variable derivatives
	:param n: length of time series
	:param sampling: the sampling rate
	:param number_of_variables: the number of variables returned, not including the input (default 1)
	:param warmup_time: the time for relaxing to the stationary regime (default 1000)
	:param tau: noise correlation time (default 3.0)
	:param eps: noise strength (default 0.5)
	:param dt: time step (default 0.01)
	:returns: time series of the signal and driving noise"""
	# initial conditions
	state = [random.gauss(0,0.2) for i in range(len(ders))]
	state[0] = state[0] + 1.0
	resS = [[] for i in range(len(ders)+1)] # +1 for driving signal
	I = 0.0
	# warmup
	for i in range(round(warmup_time/dt)):
		I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		state = one_step_integrator(state, ders, I, dt)
	# real integration
	for i in range(n*sampling):
		I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		state = one_step_integrator(state, ders, I, dt)
		for c in range(len(ders)):
			resS[c].append(state[c])
		resS[-1].append(I)
	# return list
	rl = [resS[i][::sampling] for i in range(number_of_variables)]
	rl.append(resS[-1][::sampling])
	return rl


def generate_signal_e(ders, n, sampling, number_of_variables=1, warmup_time=1000.0, tau=3.0, eps=0.5, e_eps=1.0, dt=0.01):
	"""generates signal for the oscillator driven by e^(correlated noise)
	
	:param ders: a list of state variable derivatives
	:param n: length of time series
	:param sampling: the sampling rate
	:param number_of_variables: the number of variables returned, not including the input (default 1)
	:param warmup_time: the time for relaxing to the stationary regime (default 1000)
	:param tau: noise correlation time (default 3.0)
	:param eps: noise strength (default 0.5)
	:param e_eps: noise e strenght (default 1.0)
	:param dt: time step (default 0.01)
	:returns: time series of the signal and driving noise"""
	# initial conditions
	state = [random.gauss(0,0.2) for i in range(len(ders))]
	state[0] = state[0] + 1.0
	resS = [[] for i in range(len(ders)+1)] # +1 for driving signal
	It = 0.0
	# warmup
	for i in range(round(warmup_time/dt)):
		It = It - (It/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		I = e_eps*exp(It)
		state = one_step_integrator(state, ders, I, dt)
	# real integration
	for i in range(n*sampling):
		It = It - (It/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		I = e_eps*exp(It)
		state = one_step_integrator(state, ders, I, dt)
		for c in range(len(ders)):
			resS[c].append(state[c])
		resS[-1].append(I)
	# return list
	rl = [resS[i][::sampling] for i in range(number_of_variables)]
	rl.append(resS[-1][::sampling])
	return rl
