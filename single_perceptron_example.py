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


## Questions:
# Where to the weights come from?
# Where to the biaes come from?

# np.dot(weights, x) + bias == 0

np.dot(weights, x) + bias == 0