from instances import *
from random    import gauss as gaussRandom
import numpy as np
import math

class Network:
    def __init__(self, layers):
        '''Build a new Network from a list of Layers'''
        self.layers        = layers
        self.initWtsMean   = None
        self.initWtsStdDev = None

    def __str__(self):
        '''Get a nice string representation of this Network object'''
        return ''.join([''.join(["--- LAYER ", str(x), " ---\n",
            str(self.layers[x]), "\n"]) for x in range(len(self.layers))])

    def layer(self, layerNo):
        '''Get the layer of the given number'''
        return self.layers[layerNo]

    @property
    def outputLayer(self):
        '''Get the output layer of this Network'''
        return self.layers[len(self.layers) - 1]

    @property
    def hiddenLayers(self):
        '''Get the hidden layers of this Network'''
        return self.layers[:len(self.layers) - 1]

    @staticmethod
    def zeroWtsNet(dims, layout):
        '''Create a network from the given layout, for data with dimensionality
        given by the dims parameter. A layout is a list of integers that
        specifies the network's configuration, e.g. a layout of [2, 9, 4, 1]
        will produce a network with an input layer with two nodes, two hidden
        layers, the first with 9 nodes and the second with 4, and an output
        layer with one node. All weights are initialised to zero.'''
        return Network([Layer.zeroWtsLayer(dims if x == 0 else layout[x-1],
            layout[x]) for x in range(len(layout))])

    @staticmethod
    def gaussWtsNet(mean, stdDev, dims, layout):
        '''Like zeroWtsNet, but instead of initialising all weights to zero,
        every weight (including biases) is initialised with a normally
        distributed random number drawn from a distribution with the specified
        mean and standard deviation'''
        net = Network([Layer.gaussWtsLayer(mean, stdDev,
            dims if x == 0 else layout[x-1], layout[x])
            for x in range(len(layout))])
        net.initWtsMean   = mean
        net.initWtsStdDev = stdDev
        return net

    def resetGauss(self):
        '''Reset all weights and stored delta/activation values in this network
        to freshly generated gaussian weights with the same mean and standard
        deviation as was initially used to create this Network. if gaussWtsNet()
        was not used to create this Network, then that data is not available, so
        an error is raised.'''
        for layerIndex in range(len(self.layers)):
            self.layers[layerIndex] = Layer.gaussWtsLayer(self.initWtsMean,
                    self.initWtsStdDev,
                    len(self.layers[layerIndex].nodes[0].wts) - 1,
                    len(self.layers[layerIndex].nodes))

    def fwdPass(self, inst):
        '''Given an input instance, calculate all node activations to produce an
        output vector - a prediction for the given instance'''

        vec = inst.data

        for layer in self.layers:

            for node in layer.nodes:
                node.activn = node.activation(vec)

            vec = [sigmoid(node.activn) for node in layer.nodes]

        return vec

    def trainBackProp(self, insts, rate, iters):
        '''Train the network using the back propagation algorithm, for a given
        set of instances and a given learning rate'''

        for x in range(iters):

            for inst in insts:
                # Pass the instance through the network so that node activations
                # correspond to this instance, and to get the network's output
                # for this instance
                out = self.fwdPass(inst)

                # Update delta values for the output layer
                for nodeNo in range(len(out)):
                    node = self.outputLayer.nodes[nodeNo]
                    node.delta = -1 * derivSigmoid(node.activn) * (
                            inst.label[nodeNo] - out[nodeNo])

                # Update delta values for the hidden layers
                for layerNo in range(len(self.hiddenLayers)):
                    layer = self.layers[layerNo]

                    for nodeNo in range(len(layer.nodes)):
                        node = layer.nodes[nodeNo]

                        # To calculate the value of the sum in the delta
                        # equation, iterate over every node in the next layer,
                        # calculating the product of each node's delta with the
                        # weighting it gives to this node
                        sumVal = 0
                        for toNode in self.layers[layerNo + 1].nodes:
                            sumVal = sumVal + (
                                    toNode.delta * toNode.wts[nodeNo + 1])
                                    # nodeNo + 1 because wts[0] is the bias

                        node.delta = derivSigmoid(node.activn) * sumVal

                # Update the weights for the input layer
                for node in self.layers[0].nodes:
                    node.wts[0] = node.wts[0] - (rate * node.delta)
                    for inputVal in range(len(inst.data)):
                        node.wts[inputVal + 1] = node.wts[inputVal + 1] - (
                                rate * node.delta * inst.data[inputVal])

                # Update the weights for the layers after the input layer
                for layerNo in range(1, len(self.layers)):

                    for nodeNo in range(len(self.layers[layerNo].nodes)):
                        node = self.layers[layerNo].nodes[nodeNo]

                        node.wts[0] = node.wts[0] - (rate * node.delta)
                        wtIndex = 1
                        for inputNode in self.layers[layerNo - 1].nodes:
                            node.wts[wtIndex] = node.wts[wtIndex] - (rate *
                                    node.delta * sigmoid(inputNode.activn))
                            wtIndex = wtIndex + 1

class Layer:
    def __init__(self, nodes):
        '''Build a new Layer from a list of nodes'''
        self.nodes = nodes

    def __str__(self):
        '''Get a nice string representation of this Layer object'''
        return ''.join([''.join([str(nd), "\n"]) for nd in self.nodes])

    def node(self, nodeNo):
        '''Get the node of the given number'''
        return self.nodes[nodeNo]

    @staticmethod
    def zeroWtsLayer(numWtsPerNode, numNodes):
        '''Create a single network layer for data of the specified
        dimensionality, with a number of nodes given by numNodes, and all nodes
        initialised with zero weights.'''
        return Layer([Node.zeroWtsNode(numWtsPerNode) for x in range(numNodes)])

    @staticmethod
    def gaussWtsLayer(mean, stdDev, numWtsPerNode, numNodes):
        '''Create a layer with the specified number of nodes, each with the
        specified number of weights, where the weights for each node are
        initialised with values that are each a normally distributed random
        number drawn from a distribution with the specified mean and standard
        deviation'''
        return Layer([Node.gaussWtsNode(mean, stdDev, numWtsPerNode)
            for x in range(numNodes)])

class Node:
    '''The Node class represents a single node within a neural network. A node
    object has a single member - a list of weights. The convention is that the
    first element of that list is the bias of this Node.'''

    def __init__(self, wts):
        '''Build a new Node from a weight vector'''
        self.wts    = wts
        self.delta  = None  # It is very helpful when doing backpropagation, to
        self.activn = None  # be able to store delta and activation values in
                            # the nodes that they correspond to

    def __str__(self):
        '''Get a nice string representation of this Node object'''
        return ''.join(['B: ', str(self.wts[0]), ', WTS: ', str(self.wts[1:])])

    def wt(self, wtNo):
        '''Get the weight of the given number (as usual, 0 is the index of the
        bias weight)'''
        return self.wts[wtNo]

    def setWt(self, wtNo, val):
        '''Set the weight of the given number (as usual, 0 is the index of the
        bias weight) to the given value'''
        self.wts[wtNo] = val

    @staticmethod
    def zeroWtsNode(numInputs):
        '''Create a single network node with a specified number of inputs. The
        Node will have one more weight than numInputs - the extra weight is the
        bias for this Node. Weights are initialised to 0.'''
        return Node([0 for x in range(numInputs + 1)]) # +1 for a bias weight

    @staticmethod
    def gaussWtsNode(mean, stdDev, numInputs):
        '''Create a single network node with a specified number of inputs. The
        Node will have one more weight than numInputs - the extra weight is the
        bias for this Node. Weights are initialised with normally distributed
        random values, drawn from a distribution of given mean and standard
        deviation'''
        return Node([gaussRandom(mean, stdDev) for x in range(numInputs + 1)])

    def activation(self, vec):
        '''Compute the activation of the given input vector for this node. The
        given input vector may be an instance or a vector of values from a
        previous layer. The only constraint is that the dimensionality of the
        input must match the number of weights (not including the bias) of this
        node. The retured value is the dot product of the input vector with the
        weight vector, plus the bias, fed into a sigmoid function.'''
        return np.dot(vec, self.wts[1:]) + self.wts[0]

def sigmoid(x, k=1):
    '''The sigmoid function is defined as:
        sigmoid(x) = 1 / 1 - e^(-1 * k * x)
    where 'x' is a variable and 'k' and is an optionally specified coefficient.
    The function is used when computing the activation of a Node.'''
    return 1 / (1 + (math.e ** (-1 * k * x)))

def derivSigmoid(x, k=1):
    '''Compute the value of the derivative of the sigmoid function at a given x,
    with optionally specified constant k. The derivative of the sigmoid function
    can be shown to be:
        sig'(x) = k * sig(x) * (1 - sig(x))
    where k is the same coefficient used in the sigmoid function.'''
    sigX = sigmoid(x, k)
    return k * sigX * (1 - sigX)

def euclideanDist(x, y):
    '''Compute the euclidean distance between two vectors (lists) x and y'''
    return math.sqrt(sum(map(lambda xi, yi: math.pow(xi - yi, 2), x, y)))
