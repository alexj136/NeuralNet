from instances import *
from random import gauss as gaussRandom, shuffle
import numpy as np
import math
from misc import euclideanDist

class MLPNetwork:
    def __init__(self, mean, stdDev, layout):
        '''Create a network with specified layout, where every weight (including
        biases) is initialised with a random number drawn from a normal
        distribution with the specified mean and standard deviation'''
        self.mean = mean
        self.stdDev = stdDev
        self.layout = layout
        self.reinitialise()

    def reinitialise(self):
        '''Reset the parameters - 'un-train' - this network'''
        self.layers = [Layer.gaussWtsLayer(self.mean, self.stdDev, \
                self.layout[x-1], self.layout[x]) \
                for x in range(1, len(self.layout))]

    def __str__(self):
        '''Get a nice string representation of this Network object'''
        return ''.join([''.join(["--- LAYER ", str(x), " ---\n",
            str(self.layers[x]), "\n"]) for x in range(len(self.layers))])

    def printDeltas(self):
        '''Print all nodes with their current delta values'''
        layerNo = 0
        for layer in self.layers:
            print '--- LAYER', str(layerNo), '---'
            layerNo = layerNo + 1

            nodeNo = 0
            for node in layer.nodes:
                print 'NODE', str(nodeNo), '- DELTA =', str(node.delta)
                nodeNo = nodeNo + 1

    def layer(self, layerNo):
        '''Get the layer of the given number'''
        return self.layers[layerNo]

    @property
    def outputLayer(self):
        '''Get the output layer of this Network'''
        return self.layers[len(self.layers) - 1]

    def fwdPass(self, inst):
        '''Given an input instance, calculate all node activations to produce an
        output vector - a prediction for the given instance'''

        vec = inst.data

        for layer in self.layers:

            for node in layer.nodes:
                node.activn = node.activation(vec)

            # Don't apply the sigmoid function to the output of the last layer.
            if layer is self.outputLayer:
                vec = [node.activn for node in layer.nodes]
            else:
                vec = [sigmoid(node.activn) for node in layer.nodes]

        return vec

    def train(self, insts, rate, convergenceThreshold, maxIters):
        '''Train the network using the back propagation algorithm, for a given
        set of instances and a given learning rate. Stop when the convergence
        threshold is reached, or when the maximum allowed number of iterations
        (epochs) have been reached.'''

        if convergenceThreshold is not None:
            print 'WARNING: Convergence threshold not implemented for MLP'

        for x in range(maxIters):
            
            for inst in insts:
                
                # Randomising the order in which the instances are presented 
                # reduces the risk of getting stuck in a local minimum
                shuffle(insts)

                # Pass the instance through the network so that node activations
                # correspond to this instance, and to get the network's output
                # for this instance
                out = self.fwdPass(inst)

                # Recalculate delta values for the output layer
                for nodeNo, node in enumerate(self.outputLayer.nodes):
                    node.delta = -1 * derivSigmoid(node.activn) * (
                            inst.label[nodeNo] - out[nodeNo])

                # Recalculate delta values for the hidden layers
                for layerNo in range(len(self.layers) - 2, -1, -1):
                    layer = self.layers[layerNo]

                    for nodeNo, node in enumerate(layer.nodes):
                        node.delta = derivSigmoid(node.activn) * sum([
                            (toNode.delta * toNode.wts[nodeNo + 1])
                            for toNode in self.layers[layerNo + 1].nodes])

                # Update the weights for the input layer
                for node in self.layers[0].nodes:
                    node.wts[0] = node.wts[0] - (rate * node.delta)
                    for inputVal in range(len(inst.data)):
                        node.wts[inputVal + 1] -= rate * node.delta * \
                                inst.data[inputVal]

                # Update the weights for the layers after the input layer
                for layerNo in range(1, len(self.layers)):
                    for nodeNo, node in enumerate(self.layers[layerNo].nodes):

                        node.wts[0] -= rate * node.delta
                        wtIndex = 1
                        for inputNode in self.layers[layerNo - 1].nodes:
                            node.wts[wtIndex] -= rate * node.delta * \
                                    sigmoid(inputNode.activn)
                            wtIndex += 1

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
    def gaussWtsNode(mean, stdDev, numInputs):
        '''Create a single network node with a specified number of inputs. The
        Node will have one more weight than numInputs - the extra weight is the
        bias for this Node. Weights are initialised with normally distributed
        random values, drawn from a distribution of given mean and standard
        deviation'''
        # + 1 for bias weight
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
    if   x < -100: return 0
    elif x >  100: return 1
    else         : return 1 / (1 + math.pow(math.e, -1 * k * x))

def derivSigmoid(x, k=1):
    '''Compute the value of the derivative of the sigmoid function at a given x,
    with optionally specified constant k. The derivative of the sigmoid function
    can be shown to be:
        sig'(x) = k * sig(x) * (1 - sig(x))
    where k is the same coefficient used in the sigmoid function.'''
    sigX = sigmoid(x, k)
    return k * sigX * (1 - sigX)
