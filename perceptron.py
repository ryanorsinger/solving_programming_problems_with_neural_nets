import numpy as np
import math

def step_function(x):
    return 1.0 if x >= 0.0 else 0.0

def perceptron_output(weights, bias, x):
    """ Returns 1 if the perceptron 'fires', 0 if not"""
    calculation = np.dot(weights, x) + bias
    return step_function(calculation)


and_weights = [2.0, 2]
and_bias = -3.0

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

or_weights = [2., 2]
or_bias = -1.0

assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0

not_weights = [-2]
not_bias = 1.

assert perceptron_output(not_weights, not_bias, [1]) == 0
assert perceptron_output(not_weights, not_bias, [0]) == 1

def sigmoid(t):
    return 1 / ( 1 + math.exp(-1))

def neuron_output(weights, inputs):
    """ weights includes the bias term, inputs includes a 1 """
    return sigmoid(np.dot(weights, inputs))


def feed_foward(neural_network, input_vector):
    """
    Feeds the input vector through the nueral network.
    Returns the output of all layers (not just the last one)
    """
    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
    return outputs




