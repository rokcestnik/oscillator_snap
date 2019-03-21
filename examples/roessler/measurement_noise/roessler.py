from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from oscillator_snap import *


# Roessler system
a = 0.2
b = 0.6
c = 5.7
def dx(state,I):
	return -state[1] - state[2]
def dy(state,I):
	return state[0] + a*state[1]
def dz(state,I):
	return I + b + state[2]*(state[0]-c)
ders = [dx,dy,dz]


# parameters
DATA_DIM_IN = 4
DATA_DIM_OUT = 3
PAST = 36
NODES = 32
EPOCHS = 500
LEARNING_RATE = 0.03
BATCH_SIZE = 100
DATA_SAMPLING = 17 # (this has to be chosen such that PAST*DATA_SAMPLING*DT = natural_period) 
DATA_LENGTH_PERIODS = 10000
VALIDATION_PERIODS = 25
DATA_LENGTH = (DATA_LENGTH_PERIODS+VALIDATION_PERIODS)*PAST
VALIDATION = VALIDATION_PERIODS*PAST
PLOT_RANGE = 100*PAST # plot 100 periods


# generate model
model = generate_model(DATA_DIM_IN, DATA_DIM_OUT, PAST, NODES, LEARNING_RATE, cell=LSTM, n_hidden_layers=1)
# load model
#model = load_model_dill()


# generated training data
train = generate_signal(ders, DATA_LENGTH, DATA_SAMPLING, number_of_variables=3, eps=0.1, warmup_time=3000.0, tau=3.0, dt=0.01)
# add measurement noise
for d in range(len(train)):
	for i in range(len(train[d])):
		train[d][i] = train[d][i] + random.gauss(0,1.0)
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
s_for = forecast(model, PAST, DATA_DIM_IN, x_train[0], PLOT_RANGE, osc_inp, number_of_variables=3)

# plot signal forecast
pyplot.plot([i for i in range(PLOT_RANGE)], train[0][:PLOT_RANGE])
pyplot.plot([i + PAST for i in range(PLOT_RANGE - PAST)], [s_for[i] for i in range(PLOT_RANGE - PAST)])
pyplot.savefig("prediction.png")
pyplot.show()


# plot attractor (0 input)
osc_inp = [0 for i in range(PLOT_RANGE)] # 0 input
s_for = forecast(model, PAST, DATA_DIM_IN, x_train[0], PLOT_RANGE, osc_inp, number_of_variables=3)
# plot
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot(train[0][:PLOT_RANGE], train[1][:PLOT_RANGE], train[2][:PLOT_RANGE], lw=0.5)
ax.plot([s_for[i][0] for i in range(PLOT_RANGE-PAST)], [s_for[i][1] for i in range(PLOT_RANGE-PAST)], [s_for[i][2] for i in range(PLOT_RANGE-PAST)], lw=0.5)
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.set_zlabel("z(t)")
pyplot.savefig("attractor.png")
pyplot.show()


