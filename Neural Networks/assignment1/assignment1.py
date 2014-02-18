#!/bin/python2
import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

POSITIVE = 1
NEGATIVE = -1

class Instance:
    '''
    An Instance contains:
        data  - a list of values for each dimension
        label - the class label
    '''
    def __init__(self, data, label):
        self.data  = data
        self.label = label

    def __str__(self):
        '''
        Get a nice string representation of this Instance object
        '''
        return ''.join(['DATA: ', str(self.data), ', LABEL: ', str(self.label)])

class Weights:

    def __init__(self, bias, weights):
        self.bias    = bias
        self.weights = weights

    def copy(self):
        return Weights(self.bias, [w for w in self.weights])

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
            if classify(wts, inst) != inst.label:
                converged = False
                for i in range(len(wts.weights)):
                    wts.weights[i] = wts.weights[i] + (
                            learning_rate * inst.data[i] * inst.label)
                wts.bias = wts.bias + (learning_rate * inst.label)

    return (wts, iterations)

def learn_regressor(
          training_instances
        , wts = Weights(0.0, [0.0, 0.0])
        , learning_rate = 0.1
        , iteration_cap = 100
        , convergence_threshold = 0.1
        ):
    iterations = 0
    num_insts = len(training_instances)

    while iterations < iteration_cap:

        iterations = iterations + 1

        new_wts = wts.copy()

        # Adjust the bias
        new_wts.bias = wts.bias - (learning_rate * ( \
                sum([activation(wts, inst) - inst.label \
                for inst in training_instances])/num_insts))
        
        # Adjust the weight for each dimension
        for i in range(len(wts.weights)):
            new_wts.weights[i] = wts.weights[i] - (learning_rate * ( \
                    sum([(activation(wts, inst) - inst.label) * inst.data[i] \
                    for inst in training_instances])/num_insts))


        # Conclude if the weights have converged
        if converged(wts, new_wts, convergence_threshold): break

        wts = new_wts

    return new_wts, iterations

def converged(wts1, wts2, threshold):
    '''
    Compare two weight sets to see if a weight updating algorithm has converged.
    The threshold represents an allowed degree of deviation in the given weights
    such that convergence is still indicated
    '''
    if len(wts1.weights) != len(wts2.weights):
        raise Exception(
                'Mismatched dimensions for weights & instance in converged()')
    diffs = [abs(wts1.weights[i] - wts2.weights[i]) \
            for i in range(len(wts1.weights))]
    diffs.append(abs(wts1.bias - wts2.bias))
    return max(diffs) < threshold

def classify(wts, inst):
    '''
    Get the class of an instance from the supplied weights using a heaviside
    function
    '''
    return POSITIVE if activation(wts, inst) >= 0 else NEGATIVE

def activation(wts, inst):
    '''
    The activation function for a perceptron - the dot product of the weights
    with the instance's features, plus the bias (although the bias can be seen
    as the weight on a feature that always has value 1)
    '''
    return dot(wts.weights, inst.data) + wts.bias

def dot(x, y):
    '''
    Compute the dot product of two vectors
    '''
    if len(x) is not len(y):
        raise Exception('Mismatched dimensions for weights & instance in dot()')

    return sum(map(lambda xi, yi: xi * yi, x, y))

def doPartA1():
    instance_sets = [
        [ Instance([0, 0], POSITIVE), Instance([1, 0], POSITIVE)
        , Instance([0, 1], NEGATIVE), Instance([1, 1], NEGATIVE)
        ],
        [ Instance([0, 0], POSITIVE), Instance([1, 0], NEGATIVE)
        , Instance([0, 1], POSITIVE), Instance([1, 1], NEGATIVE)
        ],
        [ Instance([0, 0], POSITIVE), Instance([1, 0], NEGATIVE)
        , Instance([0, 1], NEGATIVE), Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE), Instance([1, 0], NEGATIVE)
        , Instance([0, 1], POSITIVE), Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE), Instance([1, 0], POSITIVE)
        , Instance([0, 1], NEGATIVE), Instance([1, 1], POSITIVE)
        ],
        [ Instance([0, 0], NEGATIVE), Instance([1, 0], POSITIVE)
        , Instance([0, 1], POSITIVE), Instance([1, 1], NEGATIVE)
        ]]
    for insts in instance_sets:
        wts, iters = learn_perceptron(insts)
        for i in insts:
            print 'INST:', i.data, 'LABEL:', i.label, 'CLASS:', \
                    str(classify(wts, i))
        print wts, '\n', 'ITERATIONS: ', iters, '\n'

def doPartA2():
    # Generate random data as specified:
    posX1vals = [random.gauss(0, 1) for x in range(20)]
    posX2vals = [random.gauss(0, 1) for x in range(20)]
    negX1vals = [random.gauss(4, 1) for x in range(20)]
    negX2vals = [random.gauss(0, 1) for x in range(20)]
    insts = []
    for i in range(20):
        insts.append(Instance([posX1vals[i], posX2vals[i]], POSITIVE))
        insts.append(Instance([negX1vals[i], negX2vals[i]], NEGATIVE))

    # Train the perceptron
    wts, iterations = learn_perceptron(insts)

    # Generate two points on the decision boundary that we can use to draw the
    # line
    linex2s = [3.0, -3.0]
    linex1s = [-1 * (wts.bias + (x2 * wts.weights[1]))/wts.weights[0]
            for x2 in linex2s]

    print 'Decision boundary learnt in', iterations, 'iterations'

    # Plot the points & line
    plot.plot(posX1vals, posX2vals, 'r^', negX1vals, negX2vals, 'g^',
            linex1s, linex2s, linewidth=1.0)
    plot.show()

def doPartB1():
    '''
    For this regression task, we think of our data as 1-dimensional, and the
    second dimension is essentially our class label.
    '''

    # y = 0.4*x + 3 + delta, delta = uniform random from -10 to +10
    data = [[x, (0.4 * x) + 3 + random.uniform(-10.0, 10.0)]
            for x in range(1, 200, 2)]
    insts = [Instance([d[0]], d[1]) for d in data]
    wts, iters = learn_regressor(insts
            , wts=Weights(0.0, [0.0])
            , learning_rate = 0.00012
            , iteration_cap = 30000
            , convergence_threshold = 0.000003
            )
    print wts, ', ITERS:', iters

    # Derive 2 points that lie on our learned regressorso that we can plot the
    # line
    linex1s = [0, 200]
    linex2s = [activation(wts, Instance([x1], None)) for x1 in linex1s]
    plot.plot([d[0] for d in data], [d[1] for d in data], 'b^',
            linex1s, linex2s, linewidth=1.0)
    plot.show()

def doPartB2():
    # y = 0.4*x + 3 + delta, delta = uniform random from -10 to +10
    data = [[x1, x2, (0.4 * x1) + (1.4 * x2) + random.uniform(-100.0, 100.0)]
            for x1 in range (1, 200, 20) for x2 in range(1, 200, 20)]
    insts = [Instance([d[0], d[1]], d[2]) for d in data]

    wts, iters = learn_regressor(insts
            , wts=Weights(0.0, [0.0, 0.0])
            , learning_rate = 0.00000012
            , iteration_cap = 30000
            , convergence_threshold = 0.000003
            )
    print wts, 'ITERS:', iters

    # Derive 2 points that lie on our learned regressorso that we can plot the
    # line
    planeX1s = range(0, 200, 20)
    planeX2s = range(0, 200, 20)
    planeX2s, planeX2s = np.meshgrid(planeX1s, planeX2s)
    planeYs = [activation(wts, Instance([planeX1s[i], planeX2s[i]], None))
            for i in range(len(planeX1s))]
    fig = plot.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.scatter([i.data[0] for i in insts], [i.data[1] for i in insts],
            [i.label for i in insts])
    axis.plot_wireframe(planeX1s, planeX2s, planeYs)
    plot.show()

if __name__ == "__main__":
    #doPartA1()
    #doPartA2()
    #doPartB1()
    doPartB2()
