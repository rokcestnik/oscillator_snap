from matplotlib import pyplot

from oscillator_snap import *


# Roessler system
a = 0.2
b = 2.00
c = 5.7
def dx(state,I):
	return -state[1] - state[2]
def dy(state,I):
	return state[0] + a*state[1]
def dz(state,I):
	return I + b + state[2]*(state[0]-c)
ders = [dx,dy,dz]


# parameters
DATA_DIM_IN = 2
DATA_DIM_OUT = 1
PAST = 36
NODES = 32
EPOCHS = 500
LEARNING_RATE = 0.005
BATCH_SIZE = 100
DATA_SAMPLING = 17 # (this has to be chosen such that PAST*DATA_SAMPLING*DT = natural_period) 
DATA_LENGTH_PERIODS = 1000
VALIDATION_PERIODS = 15
DATA_LENGTH = (DATA_LENGTH_PERIODS+VALIDATION_PERIODS)*PAST
VALIDATION = VALIDATION_PERIODS*PAST
PLOT_RANGE = 15*PAST # plot 15 periods


# generate model
model = generate_model(DATA_DIM_IN, DATA_DIM_OUT, PAST, NODES, LEARNING_RATE, cell=LSTM, n_hidden_layers=1)
# load model
#model = load_model_dill()


# generated training data
train = generate_signal(ders, DATA_LENGTH, DATA_SAMPLING, eps=0.5, tau=3.0)
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
#osc_inp = [0 for i in range(PLOT_RANGE)] # 0 input
s_for = forecast(model, PAST, DATA_DIM_IN, x_train[0], PLOT_RANGE, osc_inp)

# plot signal forecast
pyplot.plot([i for i in range(PLOT_RANGE)], train[0][:PLOT_RANGE])
pyplot.plot([i + PAST for i in range(PLOT_RANGE - PAST)], [s_for[i] for i in range(PLOT_RANGE - PAST)])
pyplot.savefig("prediction.png")
pyplot.show()


# PRC estimation
f_starter = forecast_starter(generate_signal(ders, 500, DATA_SAMPLING, eps=0.5, tau=3.0), PAST, DATA_DIM_IN)
PRC = PRC_measure(model, PAST, DATA_DIM_IN, f_starter, phase_repeats=20)
PRC_true = oscillator_PRC(ders)
PRC_true[1] = [PRC_true[1][i]*DATA_SAMPLING for i in range(len(PRC_true[1]))] # just putting it in the same units

# plot PRC
pyplot.plot(PRC_true[0], PRC_true[1])
pyplot.plot(PRC[0], PRC[1])
pyplot.savefig("PRC.png")
pyplot.show()

