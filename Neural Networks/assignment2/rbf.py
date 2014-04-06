import math
from kmeans import kMeans
from mlp    import Node
from misc   import euclideanDist, meanInst

class RBFNetwork:
    def __init__(self, numProtos, numOutputs):
        '''Create a new RBFNetwork'''
        self.rbfNodes = [None for x in range(numProtos)]
        self.wtSumNodes = [None for x in range(numOutputs)]

    @property
    def isTrained(self):
        '''Determine whether or not this RBFNN has been trained.'''

    def fwdPass(self, inst):
        '''Given an input instance, calculate all node activations to produce an
        output vector - a prediction for the given instance'''
        rbfNodeOutputs = [rbfNode.activation(inst) for rbfNode in self.rbfNodes]
        return [node.activation(rbfNodeOutputs) for node in self.wtSumNodes]

    def train(wtMean, wtStdDev, insts):
        '''Train this RBFNN - calculate beta values for each RBF node, and
        perform gradient descent to learn weights for the weighted sum nodes.
        The wtMean and wtStdDev parameters are the mean and standard deviation
        of the gaussian distribution from which initial weights for the weighted
        sum nodes will be randomly drawn.'''

        self.wtSumNodes = [gaussWtsNode(wtMean, wtStdDev, len(protos))
                for x in range(len(self.wtSumNodes))]

        protos, clusters = kMeans(len(self.rbfNodes), insts)

        # Calculate beta coefficients
        betas = []
        for cluster in clusters:
            clusterMean = meanInst(cluster)
            meanDists = [euclideanDist(inst, clusterMean) for inst in cluster]
            sigma = sum(meanDists)/len(cluster)
            betas.append(1.0 / (2 * math.pow(sigma, 2)))

        # Create the RBF nodes from the prototype & beta coefficient
        self.rbfNodes = [RBFNode(proto, beta)
                for proto, beta in zip(protos, betas)]

        raise Exception('Weight learning not yet implemented')

class RBFNode:
    def __init__(self, proto, beta):
        self.proto = proto
        self.beta = beta

    def activation(self, inst):
        '''The phi function - a gaussian activation function is used here, with
        its mean at the prototype for this instance.'''
        return math.pow(math.e, -1 * self.beta * \
                math.pow(euclideanDist(inst, self.proto), 2))
