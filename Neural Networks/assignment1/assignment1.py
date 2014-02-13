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

def learn_perceptron(training_instances,
        wts = Weights(0.0, [0.0, 0.0]), learning_rate = 0.1):
    '''
    Learn a perceptron from the given training instances. If the default weights
    are used, then the given instances must be 2-dimensional.
    '''

    converged = False

    while not converged:
        
        converged = True
        erroneous_insts = [inst for inst in
                training_instances if classify(wts, inst) is not inst.label]

        if len(erroneous_insts) <= 0:
            converged = False
            errors = error(wts, erroneous_insts)
            wts.bias = wts.bias - (learning_rate * errors.bias)
            for i in range(len(wts.weights)):
                wts.weights[i] = wts.weights[i] - (learning_rate * errors.weights[i])

    return wts

def error(wts, insts):
    '''
    The Perceptron Criterion error function
    '''
    bias_err = error_single_dimension(wts.bias, [1 for i in insts], labels)
    wts_errs  = []
    for i in range(wts.length()):
        dimension_values = []
        labels           = []
        for j in range(len(insts)):
            dimension_values.append(insts[i].data[j])
            labels.append(insts[i].label)
        errors.append(error_single_dimension(wts.weights[i], dimension_weight, labels))
    return Weights(bias_err, wts_errs)

def error_single_dimension(dimension_weight, dimension_values, labels):
    '''
    The Perceptron Criterion error function for a single dimension
    '''
    if len(dimension_values) is not len(labels):
        raise Exception('Mismatched number of instances and labels in error()')

    total = 0
    for i in range(len(dimension_values)):
        total = total + (labels[i] * dimension_values[i] * dimension_weight)
    return total


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

if __name__ == "__main__":
    insts = [ Instance([0, 0], POSITIVE)
        , Instance([1, 0], NEGATIVE)
        , Instance([0, 1], POSITIVE)
        , Instance([1, 1], NEGATIVE)
        ]
    wts = learn_perceptron(insts)
    print "BIAS: ", wts.bias, ", WEIGHTS: ", wts.weights
