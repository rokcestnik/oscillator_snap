OscillatorSnap
==============

Make a snapshot of an oscillator through its time series. 

OscillatorSnap uses 
`TensorFlow <https://www.tensorflow.org/>`_
and
`Keras <https://keras.io/>`_
and provides straightforward and non-technical high-level functions meant to appeal to non-experts of artificial neural networks. 
It helps you train a recurrent neural network on oscillatory signals. 
And then from the trained network forecast the future state or probe the network for dynamical responses, e.g. estimate the phase response curve and maximal Lyapunov exponent. 


|

Citing OscillatorSnap:
.......................
If you use OscillatorSnap in your research, please cite our publication:


| Rok Cestnik and Markus Abel, Inferring the dynamics of oscillatory systems using recurrent neural networks, Chaos (2019).


+---------------------------------------------------------------------------------------------------+
|| @article{cestnik_inferring_2019,                                                                 |
|| 	author   = {R. Cestnik and M. Abel},                                                        |
|| 	title    = {Inferring the dynamics of oscillatory systems using recurrent neural networks}, |
|| 	year     = {2019},                                                                          |
|| 	journal  = {Chaos},                                                                         |
|| 	volume   = {X},                                                                             |
|| 	pages    = {X}                                                                              |
|| }                                                                                                |
+---------------------------------------------------------------------------------------------------+


|

Installing:
......................

Install with: 

.. code:: bash

	sudo pip install oscillator_snap

or download the repository and execute

.. code:: bash

	sudo python setup.py install

in its directory.


|

Simple example walkthrough: 
...........................

Make sure to import oscillator_snap (it imports everything it needs):

.. code:: python

	from oscillator_snap import *


Let's suppose we have a timeseries s(t) in the format:

.. code:: python

	data = [[s(t_1), s(t_2), s(t_3),...], [p(t_1), p(t_2), p(t_3),...]]

if one wants the signal to be generated with an ordinary differential equation this is done with:

.. code:: python

	data = generate_signal(derivatives, DATA_LENGTH, DATA_SAMPLING)

the, :code:`DATA_SAMPLING` is the ratio between the integration timestep and the timestep associated with :code:`data`, e.g. if the equation is integrated with :code:`dt = 0.01` and :code:`DATA_SAMPLING = 10` 
then the :code:`data` is sampled with a timestep of :code:`0.1`. 


There are some other parameters that need to be determined: 

.. code:: python 

	PAST = 30 # determines the number of rolls of the RNN, how many historical values are considered for the one-step prediction
	NODES = 25 # number of nodes in each layer
	LEARNING_RATE = 0.01 
	BATCH_SIZE = 100 # how many training points are fed into the network at once
	EPOCHS = 10 # how many times is the data presented to the network during training
	VALIDATION_POINTS = 200 # how many data points are going to be separated for the validation set
	
the dimensions of the input and output also have to be specified, in this example:

.. code:: python

	DIM_IN = 2
	DIM_OUT = 1

Then the validation set is separated from the training data:

.. code:: python
	
	train_data = [data[i][:-VALIDATION_POINTS] for i in range(len(data))]
	val_data = [data[i][-VALIDATION_POINTS] for i in range(len(data))]

and then the data can be parsed:

.. code:: python

	X, Y = parse_train_data(train_data, PAST, DIM_IN, DIM_OUT)
	X_val, Y_val = parse_train_data(val_data, PAST, DIM_OUT, DIM_OUT)

A model needs to be created, it can be either freshly generated:

.. code:: python

	model = generate_model(DIM_IN, DIM_OUT, PAST, NODES, LEARNING_RATE, cell=LSTM, n_hidden_layers=1)

or loaded from previous use:

.. code:: python

	model = load_model_dill()

(see further down on how to save a model). 

The model needs to be compiled:

.. code:: python

	model = compile_model(model, LEARNING_RATE)


and then it can be trained:

.. code:: python

	model.fit(X, Y, batch_size=BATHC_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val))

Once the model is trained it can be used for signal forecasting:

.. code:: python 

	inp = [0 for i in range(PLOT_RANGE)] # input
	f_starter = forecast_starter(data, PAST, DIM_IN) # initial state, to start from a different state just change 'data'
	signal_forecast = forecast(model, PAST, DIM_IN, f_starter, PLOT_RANGE, inp)


estimating the natural period, phase response curve, the maximal Lyapunov exponent and the bifurcation diagram: 

.. code:: python

	period = period_measure(model, PAST, DIM_IN, f_starter, constant_input_offset=0, thr=0.0)
	lyapunov = lyapunov_measure(model, PAST, DIM_IN, f_starter, constant_input_offset=0)
	PRC = PRC_measure(model, PAST, DIM_IN, f_starter, constant_input_offset=0.0, thr=0.0, phase_repeats=20, stimulation=1.0)
	bif = bifurcation_diagram(model, PAST, DIM_IN, f_starter, ci_min=0.1, ci_max=2.0, dci=0.005, time_window=1000)

and if the true dynamical equations are known, these quantities can be determined from equations as well for comparison:

.. code:: python

	period_eq = oscillator_period(derivatives, inp=0, thr=0.0) # if the system is chaotic the average period can be computed: 'oscillator_average_period()'
	lyapunov_eq = oscillator_lyapunov(derivatives, inp=0)
	PRC_eq = oscillator_PRC(derivatives, inp=0, thr=0.0)
	bif_eq = oscillator_bifurcation(derivatives, inp_min=0.05, inp_max=2.0, d_inp=0.01, time_window=2000)


The model as well as any objects can be saved as:

.. code:: python

	save_model_dill(model)
	save_object_dill(PRC, 'PRC')
	save_object_dill(bif, 'bifurcation')



To plot the signal, phase response curve, bifurcation...:

.. code:: python

	from matplotlib import pyplot
	
	pyplot.plot(signal_forecast[:PLOT_RANGE])
	pyplot.show()
	
	pyplot.plot(PRC[0], PRC[1])
	pyplot.show()

	pyplot.scatter(bif[0], bif[1], s=1.5)
	pyplot.show()



Other examples are found in :code:`/oscillator_snap/examples/`.




