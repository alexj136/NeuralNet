from instances import *
from random import gauss as gaussRandom

class Network:
    def __init__(self, layers):
        '''Build a new Network from a list of Layers'''
        self.layers = layers

    def __str__(self):
        '''Get a nice string representation of this Network object'''
        return ''.join([''.join(["--- LAYER ", str(x), " ---\n",
            str(self.layers[x]), "\n"]) for x in range(len(self.layers))])

    def layer(self, layerNo):
        '''Get the layer of the given number'''
        return self.layers[layerNo]

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
        return Network([Layer.gaussWtsLayer(mean, stdDev,
            dims if x == 0 else layout[x-1], layout[x])
            for x in range(len(layout))])

    def fwdPass(inst):
        '''Given an input instance, apply the network to produce an output
        vector'''
        pass

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
        self.wts = wts

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

if __name__ == '__main__':
    myNet = Network.gaussWtsNet(0, 0.3, 13, [3, 2, 1])
    print myNet
    print myNet.layer(1).node(1).wt(1)
    myNet.layer(0).node(1).setWt(1, 12)
    print myNet
