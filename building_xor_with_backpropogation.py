import numpy as np
import math
import tqdm
import random

random.seed(2)

def sigmoid(t):
    return 1 / ( 1 + math.exp(-1))

def neuron_output(weights, inputs):
    """ weights includes the bias term, inputs includes a 1 """
    return sigmoid(np.dot(weights, inputs))

def feed_forward(neural_network, input_vector):
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
    hidden_outputs, outputs = feed_forward(network, input_vector)

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
    hidden_gradients = [[hidden_deltas[i] * input for input in input_vector + [1]]
                        for i, hidden_neuron in enumerate(network[0])]

    return [hidden_gradients, output_gradients]


def gradient_step(v, gradient, step_size):
    """ 
    Moves the step_size in the gradient direction from v
    """
    assert len(v) == len(gradient), "the vector and the graidents must be the same length"

    v = np.array(v)
    gradient = np.array(gradient)

    step = step_size * gradient
    
    return v + step


# inputs
xs = [[0.0, 0.0], [0.0, 1], [1, 0.0], [1.0, 1]]

# desired outputs that match each input
ys = [[0.0], [1.0], [1.0], [0.0]]

learning_rate = 1.0

# Always start the network with random weights

network = [ # hidden layer: 2 inputs -> 2 outputs
            [[random.random() for _ in range(2 + 1)], # 1st hidden neuron
             [random.random() for _ in range(2 + 1)]],  #2nd hidden neuron
            #output layer: 2 inputs -> 1 output
            [[random.random() for _ in range(2 + 1)]] # 1st output neuron
          ]



for epoch in tqdm.trange(20000, desc="neural net for XOR gate"):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)

        # Take a gradient step for each neuron in each layer
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]

## Check that we have a trained network for XOR
assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01


