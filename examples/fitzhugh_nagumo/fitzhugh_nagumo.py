from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from oscillator_snap import *


# Fitzhugh-Nagumo excitable system
s = 0.1
a = 0.7
b = 0.8
I0 = 0.25
def dx(state,I):
	return state[0] - state[0]**3/3 - state[1] + I0 + I
def dy(state,I):
	return s*(state[0] + a - b*state[1])
ders = [dx,dy]


# parameters
DATA_DIM_IN = 2
DATA_DIM_OUT = 1
PAST = 36
NODES = 32
EPOCHS = 500
LEARNING_RATE = 0.005
BATCH_SIZE = 100
DATA_SAMPLING = 20 # (this has to be chosen such that PAST*DATA_SAMPLING*DT = natural_period) 
DATA_LENGTH_PERIODS = 50000
VALIDATION_PERIODS = 15
DATA_LENGTH = (DATA_LENGTH_PERIODS+VALIDATION_PERIODS)*PAST
VALIDATION = VALIDATION_PERIODS*PAST
PLOT_RANGE = 50*PAST # plot 50 periods
SPIKE_RANGE = 500*PAST # use 500 spikes to measure the spike train distance


# generate model
model = generate_model(DATA_DIM_IN, DATA_DIM_OUT, PAST, NODES, LEARNING_RATE)
# load model
#model = load_model_dill()


# generated training data
train = generate_signal(ders, DATA_LENGTH, DATA_SAMPLING, warmup_time=10000.0, eps=0.05, tau=25.0, dt=0.05)
# validation set separate
val = train[:]
for i in range(len(train)):
	train[i] = train[i][:-VALIDATION]
	val[i] = train[i][-VALIDATION:]
		
x_train, y_train = parse_train_data(train, PAST, DATA_DIM_IN, DATA_DIM_OUT)
x_val, y_val = parse_train_data(val, PAST, DATA_DIM_IN, DATA_DIM_OUT)


# compile model
model = compile_model(model, LEARNING_RATE)
# train
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val))


# save model with h5py file
save_model_h5py(model)
# save model manually with dill
save_model_dill(model)


# forecast
osc_inp = [x_train[i][-1][1] for i in range(PLOT_RANGE)] # original input
#osc_inp = [2 for i in range(PLOT_RANGE)] # different input
s_for = forecast(model, PAST, DATA_DIM_IN, x_train[0], PLOT_RANGE, osc_inp)

# plot signal forecast
pyplot.plot([i for i in range(PLOT_RANGE)], train[0][:PLOT_RANGE])
pyplot.plot([i + PAST for i in range(PLOT_RANGE - PAST)], [s_for[i][0]+4.0 for i in range(PLOT_RANGE - PAST)]) # +4 is just for visual distinction of signals
pyplot.savefig("prediction_train_input.png")
pyplot.show()


# forecast with a different realization of the same input
train = generate_signal(ders, round(DATA_LENGTH/10.0), DATA_SAMPLING, warmup_time=10000.0, eps=0.05, tau=25.0, dt=0.05)
x_train, y_train = parse_train_data(train, PAST, DATA_DIM_IN, DATA_DIM_OUT)
osc_inp = [x_train[i][-1][1] for i in range(PLOT_RANGE)]
s_for = forecast(model, PAST, DATA_DIM_IN, x_train[0], PLOT_RANGE, osc_inp)

# plot signal forecast
pyplot.plot([i for i in range(PLOT_RANGE)], train[0][:PLOT_RANGE])
pyplot.plot([i + PAST for i in range(PLOT_RANGE - PAST)], [s_for[i][0]+4.0 for i in range(PLOT_RANGE - PAST)]) # +4 is just for visual distinction of signals
pyplot.savefig("prediction_test_input.png")
pyplot.show()
