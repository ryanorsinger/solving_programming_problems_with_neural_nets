import numpy as np
import math

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
