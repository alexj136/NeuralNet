#!/bin/python2
# Candidate No: 18512
import random, math
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

def grad_desc_sequential(training_instances, wts
        , learning_rate = 0.1
        , iteration_cap = 100
        , collect_weights = False
        ):
    '''
    Learn a perceptron from the given training instances. If the default weights
    are used, then the given instances must be 2-dimensional. The algorithm used
    is sequential gradient descent.
        If collect_weights is set, the weights will be recorded at each epoch
    and returned in a list, rather than just returning the final weights.
    '''

    if collect_weights:
        weights_list = []

    converged = False
    iterations = 0
    while not converged and iterations < iteration_cap:

        if collect_weights: weights_list.append(wts.copy())

        converged = True
        iterations = iterations + 1

        for inst in training_instances:
            if heaviside_classify(wts, inst) != inst.label:
                converged = False
                for i in range(len(wts.weights)):
                    wts.weights[i] = wts.weights[i] + (
                            learning_rate * inst.data[i] * inst.label)
                wts.bias = wts.bias + (learning_rate * inst.label)

    return (weights_list if collect_weights else wts, iterations)

def grad_desc_batch(training_instances, wts
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

def heaviside_classify(wts, inst):
    '''
    Get the class of an instance from the supplied weights using a heaviside
    function
    '''
    return POSITIVE if activation(wts, inst) >= 0 else NEGATIVE

def sigmoid_classify(wts, inst, coefficient=1):
    '''
    Derive a value that represents both a class (+1 for values greater than 0,
    -1 for values less than 0) and a degree of certainty for our class decision.
    The more extreme the value, the more certain our decision is. The closer to
    0 the value is, the less certain we are.
        A sigmoidal function S(t) = 1 / 1 - e^-ax, where a is the (optionally)
    given coefficient, and x is the dot product of the given weight & instance,
    is used.
    '''
    return 1 / (1 + (math.e ** (-1 * coefficient * activation(wts, inst))))

def tanh_classify(wts, inst):
    '''
    This function uses the hyperbolic tangent function to obtain a degree of
    certainty for classification, as with sigmoid_classify
    '''
    return math.tanh(activation(wts, inst))


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

    print '--- SEQUENTIAL ---'
    for insts in instance_sets:
        seq_wts, seq_iters = grad_desc_sequential(insts,
                Weights(0.0, [0.0, 0.0]))
        for i in insts:
            print 'INST:', i.data, 'LABEL:', i.label, 'CLASS:', \
                    str(heaviside_classify(seq_wts, i))
        print seq_wts, '\n', 'ITERATIONS: ', seq_iters, '\n'

    print '--- BATCH ---'
    for insts in instance_sets:
        bat_wts, bat_iters = grad_desc_batch(insts, Weights(0.0, [0.0, 0.0]),
                convergence_threshold=0.01)
        for i in insts:
            print 'INST:', i.data, 'LABEL:', i.label, 'CLASS:', \
                    str(heaviside_classify(bat_wts, i))
        print bat_wts, '\n', 'ITERATIONS: ', bat_iters, '\n'

    # Illustrate procedure with instance_sets[4] using sequential by showing the
    # graph after each iteration
    wts_list, iters = grad_desc_sequential(instance_sets[3],
            Weights(0.0, [0.0, 0.0]), collect_weights=True)
    negX1vals = [i.data[0] for i in instance_sets[3] if i.label is NEGATIVE]
    negX2vals = [i.data[1] for i in instance_sets[3] if i.label is NEGATIVE]
    posX1vals = [i.data[0] for i in instance_sets[3] if i.label is POSITIVE]
    posX2vals = [i.data[1] for i in instance_sets[3] if i.label is POSITIVE]
    iter_num = 0
    for wts in wts_list:
        iter_num = iter_num + 1
        print 'ITERATION', iter_num, 'OF', iters, ':', wts
        # Generate two points on the decision boundary that we can use to draw
        # the line
        linex1s = [-0.2, 1.2]
        linex2s = [-1 * (wts.bias + (x1 * wts.weights[1]))/(wts.weights[0] \
                if wts.weights[0] != 0 else 0.000000001) for x1 in linex1s]

        # Plot the points & line
        plot.plot(posX2vals, posX1vals, 'r^', negX2vals, negX1vals, 'g^',
                linex1s, linex2s, linewidth=1.0)
        plot.xlabel('X1')
        plot.ylabel('X2')
        plot.ylim([-0.2, 1.2])
        plot.xlim([-0.2, 1.2])
        plot.show()

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
    wts, iterations = grad_desc_sequential(insts, Weights(0.0, [0.0, 0.0]))

    # Generate two points on the decision boundary that we can use to draw the
    # line
    linex2s = [3.0, -3.0]
    linex1s = [-1 * (wts.bias + (x2 * wts.weights[1]))/wts.weights[0]
            for x2 in linex2s]

    print 'Decision boundary learnt in', iterations, 'iterations'

    # Plot the points & line
    plot.plot(posX1vals, posX2vals, 'r^', negX1vals, negX2vals, 'g^',
            linex1s, linex2s, linewidth=1.0)
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.show()
    """
    # Illustrate procedure with instance_sets[4] using sequential by showing the
    # graph after each iteration
    wts_list, iters = grad_desc_sequential(insts,
            Weights(0.0, [0.0, 0.0]), collect_weights=True)
    iter_num = 0
    for wts in wts_list:
        iter_num = iter_num + 1
        print 'ITERATION', iter_num, 'OF', iters, ':', wts
        # Generate two points on the decision boundary that we can use to draw
        # the line. Prevent division by zero by substituting zero values with a
        # very small number, that produces the same line.
        linex2s = [3.0, -3.0]
        linex1s = [-1 * (wts.bias + (x2 * wts.weights[1]))/(wts.weights[0]
                if wts.weights[0] != 0 else 0.000000001) for x2 in linex2s]

        # Plot the points & line
        plot.plot(posX1vals, posX2vals, 'r^', negX1vals, negX2vals, 'g^',
                linex1s, linex2s, linewidth=1.0)
        plot.xlabel('X1')
        plot.ylabel('X2')
        plot.show()
    """

def doPartB1():
    '''
    For this regression task, we think of our data as 1-dimensional, and the
    second dimension is essentially our class label.
    '''
    # y = 0.4*x + 3 + delta, delta = uniform random from -10 to +10
    data = [[x, (0.4 * x) + 3 + random.uniform(-10.0, 10.0)]
            for x in range(1, 200, 2)]
    insts = [Instance([d[0]], d[1]) for d in data]
    wts, iters = grad_desc_batch(insts
            , wts=Weights(0.0, [0.0])
            , learning_rate = 0.00012
            , iteration_cap = 70000
            , convergence_threshold = 0.000059
            )
    print wts, ', ITERS:', iters

    # Derive 2 points that lie on our learned regressorso that we can plot the
    # line
    linex1s = [0, 200]
    linex2s = [activation(wts, Instance([x1], None)) for x1 in linex1s]
    plot.plot([d[0] for d in data], [d[1] for d in data], 'b^',
            linex1s, linex2s, linewidth=1.0)
    plot.xlabel('X')
    plot.ylabel('Label')
    plot.show()

def doPartB2():
    # y = 0.4*x1 + 1.4*x2 + delta, delta = uniform random from -100 to +100
    data = [[x1, x2, (0.4 * x1) + (1.4 * x2) + random.uniform(-100, 100)]
            for x1 in range (1, 200, 20) for x2 in range(1, 200, 20)]
    insts = [Instance([d[0], d[1]], d[2]) for d in data]

    wts, iters = grad_desc_batch(insts
            , wts=Weights(0.0, [0.0, 0.0])
            , learning_rate = 0.00000012
            , iteration_cap = 30000
            , convergence_threshold = 0.000003
            )
    print wts, 'ITERS:', iters

    # Derive some points that lie on our regressor plane, such that we can draw
    # the plane nicely
    planeX1s = range(0, 200, 20)
    planeX2s = range(0, 200, 20)
    planeX1s, planeX2s = np.meshgrid(planeX1s, planeX2s)
    planeYs = [activation(wts, Instance([planeX1s[i], planeX2s[i]], None))
            for i in range(len(planeX1s))]
    fig = plot.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.scatter([i.data[0] for i in insts], [i.data[1] for i in insts],
            [i.label for i in insts])
    axis.plot_wireframe(planeX1s, planeX2s, planeYs)
    axis.set_xlabel('X1')
    axis.set_ylabel('X2')
    axis.set_zlabel('Label')
    plot.show()

    print 'av bias', str(np.mean([w.bias for w in ws])), 'av w1', str(np.mean([w.weights[0] for w in ws])), 'av w2', str(np.mean([w.weights[1] for w in ws])), 'av iters', str(np.mean(iss))

if __name__ == "__main__":
    # Uncomment as required
    #doPartA1()
    doPartA2()
    #doPartB1()
    #doPartB2()
