#!/bin/python2

POSITIVE = 1
NEGATIVE = -1

class Instance:

    def __init__(self, data, label):
        self.data  = data
        self.label = label

    def datum(self, index):
        return self.data[index]

    def length(self):
        return len(self.data)

class Weights:

    def __init__(self, bias, weights):
        self.bias    = bias
        self.weights = weights

    def bias(self):
        return self.bias

    def weight(self, index):
        return self.weights[index]

    def length(self):
        return len(self.weights)

    def __str__(self):
        '''
        Get a nice string representation of this Weights object
        '''
        return ''.join(['BIAS: ', str(self.bias), ', WEIGHTS: ',
            str(self.weights)])

def learn_perceptron(
          training_instances
        , wts = Weights(0.0, [0.0, 0.0])
        , learning_rate = 0.1
        , iteration_cap = 100
        ):
    '''
    Learn a perceptron from the given training instances. If the default weights
    are used, then the given instances must be 2-dimensional.
    '''

    converged = False
    iterations = 0
    while not converged and iterations < iteration_cap:

        converged = True
        iterations = iterations + 1

        for inst in training_instances:
            if classify(wts, inst) is not inst.label:
                converged = False
                for i in range(len(wts.weights)):
                    wts.weights[i] = wts.weights[i] + (
                            learning_rate * inst.data[i] * inst.label)
                wts.bias = wts.bias + (learning_rate * inst.label)

    return (wts, iterations)
"""
def error(wts, insts):
    '''
    The Perceptron Criterion error function
    '''
    wts_errs = [] # List to collect our error values for each dimension

    for i in range(wts.length()):
        dimension_values = []
        labels           = []

        for j in range(len(insts)):
            dimension_values.append(insts[i].data[j])
            labels.append(insts[i].label)

        wts_errs.append(error_single_dimension(wts.weights[i], dimension_values, labels))

    bias_err = error_single_dimension(wts.bias, [1 for i in insts], labels)
    return Weights(bias_err, wts_errs)

def error_single_dimension(dimension_weight, dimension_values, labels):
    '''
    The Perceptron Criterion error function for a single dimension
    '''
    if len(dimension_values) is not len(labels):
        raise Exception('Mismatched number of instances and labels in error()')

    return sum([labels [i] * dimension_values[i] * dimension_weight
        for i in range(len(dimension_values))])
"""

def classify(wts, inst):
    '''
    Get the class of an instance from the supplied weights - essentially the dot
    product of the instance & weights, plus the bias
    '''
    return POSITIVE if dot(wts.weights, inst.data) + wts.bias >= 0 else NEGATIVE

def dot(x, y):
    '''
    Compute the dot product of two vectors
    '''
    if len(x) is not len(y):
        raise Exception('Mismatched dimensions for weights & instance in dot()')

    return sum(map(lambda xi, yi: xi * yi, x, y))

def doPartA1():
    instance_sets = [
        [ Instance([0, 0], POSITIVE)
        , Instance([1, 0], POSITIVE)
        , Instance([0, 1], NEGATIVE)
        , Instance([1, 1], NEGATIVE)
        ],
        [ Instance([0, 0], POSITIVE)
        , Instance([1, 0], NEGATIVE)
        , Instance([0, 1], POSITIVE)
        , Instance([1, 1], NEGATIVE)
        ],
        [ Instance([0, 0], POSITIVE)
        , Instance([1, 0], NEGATIVE)
        , Instance([0, 1], NEGATIVE)
        , Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE)
        , Instance([1, 0], NEGATIVE)
        , Instance([0, 1], POSITIVE)
        , Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE)
        , Instance([1, 0], POSITIVE)
        , Instance([0, 1], NEGATIVE)
        , Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE)
        , Instance([1, 0], POSITIVE)
        , Instance([0, 1], POSITIVE)
        , Instance([1, 1], NEGATIVE)
        ]]
    for insts in instance_sets:
        wts, iters = learn_perceptron(insts)
        for i in insts:
            print 'INST: ', i.data, ' LABEL: ', i.label, ', CLASS: ', str(classify(wts, i))
        print wts, '\n', 'ITERATIONS: ', iters, '\n'


if __name__ == "__main__":
    doPartA1()
