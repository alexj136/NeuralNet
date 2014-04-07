import math
from copy   import deepcopy
from kmeans import kMeans
from mlp    import Node
from misc   import euclideanDist, meanInst

class RBFNetwork:
    def __init__(self):
        '''Create a new RBFNetwork'''
        self.rbfNodes = None
        self.wtSumNodes = None

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

    def train(self, wtMean, wtStdDev, insts, numProtos, rate, \
            convergenceThreshold, maxIters=1000):
        '''Train this RBFNN - calculate beta values for each RBF node, and
        perform gradient descent to learn weights for the weighted sum nodes.
        The wtMean and wtStdDev parameters are the mean and standard deviation
        of the gaussian distribution from which initial weights for the weighted
        sum nodes will be randomly drawn.'''

        protos, clusters = kMeans(numProtos, insts)

        print map(len, clusters), 'NON-EMPTY:', \
                sum([1 if len(c) != 0 else 0 for c in clusters])

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

        # Create weighted sum nodes
        self.wtSumNodes = [Node.gaussWtsNode(wtMean, wtStdDev,
                len(protos)) for x in range(len(insts[0].label))]

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

        # Perform gradient descent to learn weights for the output nodes. The
        # algorithm is run independently for each dimension of the network
        # output.
        for outputIndex, node in enumerate(self.wtSumNodes):

            # Set up the ConvergenceTester for this run of gradient descent
            conv = ConvergenceTester(convergenceThreshold)

            for x in range(maxIters):
                for inst in insts:
                    rbfLayerOut = [1] + self.passRBFLayer(inst) # [1] + for bias
                    for wtIndex in range(len(node.wts)):
                        node.wts[wtIndex] = node.wts[wtIndex] + (rate *
                                rbfLayerOut[wtIndex] * inst.label[outputIndex])

                if conv.test(node.wts): break

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
