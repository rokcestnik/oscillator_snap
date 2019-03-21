from matplotlib import pyplot

from oscillator_snap import *


# Roessler system
a = 0.2
b = 0.0
c = 5.7
def dx(state,I):
	return -state[1] - state[2]
def dy(state,I):
	return state[0] + a*state[1]
def dz(state,I):
	return I + state[2]*(state[0]-c)
ders = [dx,dy,dz]


# parameters
DATA_DIM_IN = 2
DATA_DIM_OUT = 1
PAST = 36
NODES = 32
EPOCHS = 1000
LEARNING_RATE = 0.005
BATCH_SIZE = 100
DATA_SAMPLING = 17 # (this has to be chosen such that PAST*DATA_SAMPLING*DT = natural_period) 
DATA_LENGTH_PERIODS = 10000
VALIDATION_PERIODS = 15
DATA_LENGTH = (DATA_LENGTH_PERIODS+VALIDATION_PERIODS)*PAST
VALIDATION = VALIDATION_PERIODS*PAST
PLOT_RANGE = 15*PAST # plot 15 periods


# generate model
model = generate_model(DATA_DIM_IN, DATA_DIM_OUT, PAST, NODES, LEARNING_RATE)
# load model
#model = load_model_dill()


# generated training data
train = generate_signal_e(ders, DATA_LENGTH, DATA_SAMPLING, e_eps=0.5)
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
pyplot.plot([i + PAST for i in range(PLOT_RANGE - PAST)], [s_for[i] for i in range(PLOT_RANGE - PAST)])
pyplot.savefig("prediction.png")
pyplot.show()


# period estimation
f_starter = forecast_starter(generate_signal_e(ders, 500, DATA_SAMPLING, e_eps=0.5), PAST, DATA_DIM_IN)
period_args = [0.1 + 0.01*i for i in range(int((2.0-0.1)/0.01))]
period = [DATA_SAMPLING*0.01*period_measure(model, PAST, DATA_DIM_IN, f_starter, per_arg) for per_arg in period_args]
period_true = [oscillator_average_period(ders, per_arg) for per_arg in period_args]

# plot period
pyplot.plot(period_args, period_true)
pyplot.plot(period_args, period)
pyplot.savefig("period.png")
pyplot.show()


# PRC estimation
#f_starter = forecast_starter(generate_signal_e(ders, 500, DATA_SAMPLING, e_eps=0.5), PAST, DATA_DIM_IN)
PRC = PRC_measure(model, PAST, DATA_DIM_IN, f_starter, constant_input_offset=2.0)
PRC_true = oscillator_PRC(ders, inp=2.0)
PRC_true[1] = [PRC_true[1][i]*DATA_SAMPLING for i in range(len(PRC_true[1]))] # just putting in the same units

# plot PRC
pyplot.plot(PRC_true[0], PRC_true[1])
pyplot.plot(PRC[0], PRC[1])
pyplot.savefig("PRC.png")
pyplot.show()


# bifurcation diagram
#f_starter = forecast_starter(generate_signal_e(ders, 500, DATA_SAMPLING, e_eps=0.5), PAST, DATA_DIM_IN)
bif = bifurcation_diagram(model, PAST, DATA_DIM_IN, f_starter, 0.1, 2.0, 0.005, time_window=1000)
save_object_dill(bif, "bifurcation")
bif_true = oscillator_bifurcation(ders, 0.05, 2.0, 0.01, time_window=2000)
save_object_dill(bif_true, "bifurcation_true")
# load the bifurcations (if already computed)
#bif_true = load_object_dill("bifurcation_true")
#bif = load_object_dill("bifurcation")

# bifurcation plot
pyplot.scatter(bif_true[0], bif_true[1], s=0.01)
pyplot.scatter(bif[0], bif[1], s=1.5)
pyplot.savefig("bif.png")
pyplot.show()


# Lyapunov measure
#f_starter = forecast_starter(generate_signal_e(ders, 500, DATA_SAMPLING, e_eps=0.5), PAST, DATA_DIM_IN)
lyapunov_args = [0.1 + 0.01*i for i in range(int((2.0-0.1)/0.01))]
lyapunovs = [lyapunov_measure(model, PAST, DATA_DIM_IN, f_starter, constant_input_offset=lyapunov_args[i]) for i in range(len(lyapunov_args))]
save_object_dill(lyapunovs, "lyapunovs")
lyapunov_true_args = [0.1 + 0.005*i for i in range(int((2.0-0.1)/0.005))]
lyapunovs_true = [oscillator_lyapunov(ders, inp=lyapunov_true_args[i]) for i in range(len(lyapunov_true_args))]
save_object_dill(lyapunovs_true, "lyapunovs_true")
# load lyapunovs (if already computed)
#lyapunovs_true = load_object_dill("lyapunovs_true")
#lyapunovs = load_object_dill("lyapunovs")

# Lyapunov plot
pyplot.scatter(bif_true[0], np.array(bif_true[1])/100, s=0.01)
pyplot.plot(lyapunov_true_args, np.array(lyapunovs_true), 'r')
pyplot.plot(lyapunov_args, 1.0/(DATA_SAMPLING*0.01)*np.array(lyapunovs), 'g')
pyplot.savefig("lyapunov.png")
pyplot.show()
