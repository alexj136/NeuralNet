import math
from copy   import deepcopy
from kmeans import kMeans
from mlp    import Node
from misc   import euclideanDist, meanInst, flatten

class RBFNetwork:
    def __init__(self, mean, stdDev, numProtos, numOutputs):
        '''Create a new RBFNetwork'''
        self.mean = mean
        self.stdDev = stdDev
        self.numProtos = numProtos
        self.numOutputs = numOutputs
        self.reinitialise()

    def reinitialise(self):
        '''Reset the parameters - 'un-train' - this network'''
        self.rbfNodes = None
        self.wtSumNodes = [Node.gaussWtsNode(self.mean, self.stdDev, \
                self.numProtos) for x in range(self.numOutputs)]

    @property
    def isTrained(self):
        '''Determine whether or not this RBFNN has been trained.'''
        return self.rbfNodes is None or self.wtSumNodes is None

    def passRBFLayer(self, inst):
        '''Pass an instance through the RBF layer of this RBFNetwork, to obtain
        response values for each node.'''
        return [node.activation(inst) for node in self.rbfNodes]

    def fwdPass(self, inst):
        '''Given an input instance, calculate all node activations to produce an
        output vector - a prediction for the given instance'''
        rbfNodeOutputs = self.passRBFLayer(inst)
        return [node.activation(rbfNodeOutputs) for node in self.wtSumNodes]

    def train(self, insts, rate, convergenceThreshold, maxIters):
        '''Train this RBFNN - calculate beta values for each RBF node, and
        perform gradient descent to learn weights for the weighted sum nodes.
        The wtMean and wtStdDev parameters are the mean and standard deviation
        of the gaussian distribution from which initial weights for the weighted
        sum nodes will be randomly drawn.'''

        protos, clusters = kMeans(self.numProtos, insts)

        # Filter empty clusters
        newProtos = []
        newClusters = []
        toRemove = [False if len(c) == 0 else True for c in clusters]
        for idx, shouldKeep in enumerate(toRemove):
            if shouldKeep:
                newProtos.append(protos[idx])
                newClusters.append(clusters[idx])
        protos = newProtos
        clusters = newClusters

        # Calculate beta coefficients
        betas = []
        for cluster in clusters:
            # If the cluster is empty, make the beta coefficient equal 1, which
            # will cause the activation of this node decrease very sharply as
            # the given instance gets further from the prototype, effectively
            # rendering that prototype irrelevant.
            if len(cluster) == 0:
                betas.append(0)
            else:
                clusterMean = meanInst(cluster)
                dists = [euclideanDist(inst.data, clusterMean.data)
                        for inst in cluster]
                sigma = sum(dists)/len(cluster)
                if sum(dists) == 0:
                    betas.append(1)
                else:
                    betas.append(1.0 / (2 * math.pow(sigma, 2)))

        # Create the RBF nodes from the prototype & beta coefficient
        self.rbfNodes = [RBFNode(proto, beta)
                for proto, beta in zip(protos, betas)]


        # Perform gradient descent to learn weights for the output nodes.
        conv = ConvergenceTester(convergenceThreshold)
        for x in range(maxIters):

            rbfOutputs = [[1] + self.passRBFLayer(inst) for inst in insts]
            predictions = [self.fwdPass(inst) for inst in insts]

            for outputIndex, node in enumerate(self.wtSumNodes):

                for wtIdx in range(len(node.wts)):
                    node.wts[wtIdx] -= (rate * (sum([( \
                            predictions[i][outputIndex] - \
                            inst.label[outputIndex]) * rbfOutputs[i][wtIdx] \
                            for i, inst in enumerate(insts)])/len(insts)))

            if conv.test(flatten([node.wts for node in self.wtSumNodes])): break

class RBFNode:
    def __init__(self, proto, beta):
        '''Create a new RBFNode'''
        self.proto = proto
        self.beta = beta

    def activation(self, inst):
        '''The phi function - a gaussian activation function is used here, with
        its mean at the prototype for this instance.'''
        return math.pow(math.e, -1 * self.beta * \
                math.pow(euclideanDist(inst.data, self.proto.data), 2))

class ConvergenceTester:
    '''Class used to determine if an algorithm has converged to a set of
    weights. Stores values used to test against the next value, and a threshold
    of allowed deviation between the stored and the presented value within which
    convergence is indicated.'''
    def __init__(self, threshold):
        self.threshold = threshold
        self.storedValues = None

    def test(self, values):
        '''Test to see if convergence has occured'''

        if self.storedValues is None:
            self.storedValues = deepcopy(values)
            return False

        else:
            assert len(self.storedValues) == len(values)
            result = max([abs(a - b) for a, b in \
                    zip(self.storedValues, values)]) < self.threshold
            self.storedValues = deepcopy(values)
            return result
