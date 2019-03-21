from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import copy

from oscillator_snap import *


# Mackey-Glass equation
a = 2
b = 1
n = 8
Dtau = 2
def dx(x, xtau, I):
	return a*xtau/(1+xtau**n) - b*x + I

def integrator(x, xtau, I, dt):
	k1 = dx(x,xtau, I)
	k2 = dx(x+k1/2*dt, xtau, I)
	k3 = dx(x+k2/2*dt, xtau, I)
	k4 = dx(x+k3*dt, xtau, I)
	return x + (k1+2*k2+2*k3+k4)/6*dt

def generate_MG(n, sampling, warmup_time=10000.0, tau=10.0, eps=0.05, dt=0.01):
	# history
	N = round(Dtau/dt) # how much history must be kept
	x_buf = [0 for i in range(N)] # the historical values buffer
	# initial condition
	for i in range(N): x_buf[i] = 0.5
	x_index = 0
	resS = [[] for i in range(2)] # signal and input
	I = 0.0
	# warmup
	for t in [i*dt for i in range(round(warmup_time/dt))]:
		I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		x_buf[(x_index+1)%N] = integrator(x_buf[x_index], x_buf[(x_index+1)%N], I, dt)
		x_index = (x_index+1)%N # index increase
	# real integration
	for i in range(n*sampling):
		I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		x_buf[(x_index+1)%N] = integrator(x_buf[x_index], x_buf[(x_index+1)%N], I, dt)
		x_index = (x_index+1)%N # index increase
		resS[0].append(x_buf[x_index])
		resS[1].append(I)
	# substract the verage value
	resS[0] = resS[0]-np.average(resS[0])
	return [resS[0][::sampling], resS[1][::sampling]]


# parameters
DATA_DIM_IN = 2
DATA_DIM_OUT = 1
PAST = 36
NODES = 32
EPOCHS = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 100
DATA_SAMPLING = 15 # (this has to be chosen such that PAST*DATA_SAMPLING*DT = natural_period) 
DATA_LENGTH_PERIODS = 5000
VALIDATION_PERIODS = 25
DATA_LENGTH = (DATA_LENGTH_PERIODS+VALIDATION_PERIODS)*PAST
VALIDATION = VALIDATION_PERIODS*PAST
PLOT_RANGE = 25*PAST # plot 25 periods


# generate model
model = generate_model(DATA_DIM_IN, DATA_DIM_OUT, PAST, NODES, LEARNING_RATE, cell=LSTM, n_hidden_layers=1)
# load model
#model = load_model_dill()


# generated training data
train = generate_MG(DATA_LENGTH, DATA_SAMPLING, eps=0.005, tau=1.0)
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

# plot attractor
delay = 6
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot(train[0][:PLOT_RANGE-2*delay], train[0][delay:PLOT_RANGE-delay], train[0][2*delay:PLOT_RANGE], lw=0.5)
ax.plot([s_for[i][0] for i in range(0,PLOT_RANGE-2*delay)], [s_for[i][0] for i in range(delay,PLOT_RANGE-delay)], [s_for[i][0] for i in range(2*delay,PLOT_RANGE)], lw=0.5)
ax.set_xlabel("x(t)")
ax.set_ylabel("x(t-tau)")
ax.set_zlabel("x(t-2tau)")
pyplot.savefig("attractor.png")
pyplot.show()
