import numpy as np
import theano
import theano.tensor as T
import lasagne

import ctc

num_classes = 5
mbsz = 1
min_len = 10
max_len = 10
n_hidden = 100
grad_clip = 100

input_lens = T.ivector('input_lens')
output = T.ivector('output')
output_lens = T.ivector('output_lens')

l_in = lasagne.layers.InputLayer(shape=(mbsz, max_len, num_classes))

l_forward_1 = lasagne.layers.RecurrentLayer(l_in, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.rectify)
l_forward_2 = lasagne.layers.RecurrentLayer(l_forward_1, num_classes, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh)
l_out = lasagne.layers.ReshapeLayer(l_forward_2, ((max_len, mbsz, num_classes)))

network_output = lasagne.layers.get_output(l_out)

cost = T.mean(ctc.cpu_ctc_th(network_output, input_lens, output, output_lens))
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.adam(cost, all_params, 0.001)

train = theano.function([l_in.input_var, input_lens, output, output_lens], cost, updates=updates)
predict = theano.function([l_in.input_var], network_output)

from loader import DataLoader
data_loader = DataLoader(mbsz=mbsz, min_len=min_len, max_len=max_len, num_classes=num_classes)

while True:
    sample = data_loader.sample()
    cost = train(*sample)
    out = predict(sample[0])
    print cost
    print "input", sample[0][0].argmax(1)
    print "prediction", out[:, 0].argmax(1)
    print "expected", sample[2][:sample[3][0]]



