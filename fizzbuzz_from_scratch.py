## FizzBuzz implementation from scratch

import numpy as np
import math
import tqdm

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

def sqerror_gradients(network, input_vector, target_vector):
    """
    Given a neural network, an input vector and a target vector
    Make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights
    """
    
    # Forward pass
    hidden_outputs, outputs = feed_foward(network, input_vector)

    # Gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                        for output, target in zip(outputs, target_vector)]

    # Gradients with respect to output neuron weights
    output_gradients = [[output_deltas[i] * hidden_output
                            for hidden_output in hidden_outputs + [1]]
                                for i, output_neuron in enumerate(network[-1])]

    # Gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) * np.dot(output_deltas, [n[i] for n in network[-1]])
                        for i, hidden_output in enumerate(hidden_outputs)]

    # Gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                        for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]


def gradient_step(v, gradient, step_size):
    """ 
    Moves the step_size in the gradient direction from v
    """
    assert len(v) == len(gradient), "the vector and the graidents must be the same length"

    v = np.array(v)
    gradient = np.array(gradient)

    step = step size * gradient
    
    return v + step

