from random import uniform
from instances import *
from misc import *

def minmax(insts):
    '''Return an instance with feature values that are the minimum of all
    feature and label values across the given instances. Return an instance with
    maximum values also.'''

    minFeats = []
    maxFeats = []

    for feat in range(len(insts[0].data)):
        feats = [inst.data[feat] for inst in insts]
        minFeats.append(min(feats))
        maxFeats.append(max(feats))

    return Instance(minFeats, None), Instance(maxFeats, None)

def kRandomPrototypes(k, insts):
    '''Generate k random prototypes for the given data. Prototypes do not have
    labels.'''
    minInst, maxInst = minmax(insts)
    return [Instance(
                    [uniform(minInst.data[feat], maxInst.data[feat])
                            for feat in range(len(insts[0].data))]
                    , None) for x in range(k)]

def kMeans(k, insts):
    '''Perform k-means clustering on the given instances. Return the k prototype
    instances (these are not instances from the data set itself).'''

    protos = kRandomPrototypes(k, insts)

    prevClustering = bins(k, insts)
    converged = false

    while not converged:


        # Assignment step - put each instance in the cluster corresponding to
        # its closest prototype
        newClustering = [[] for x in range(k)]
        for inst in insts:
            # Find the prototype closest to this instance and add the instance
            # to the corresponding cluster
            bestIdx = 0
            bestDist = euclideanDist(inst.data, protos[bestIdx].data)
            for idx in range(len(protos)):
                curDist = euclideanDist(inst.data, protos[idx].data)
                if curDist < bestDist:
                    bestIdx  = idx
                    bestDist = curDist
            clusters[bestIdx].append(inst)

        # Recompute the prototypes to be mean values of data in their cluster
        raise Exception('kMeans: prototype update not yet implemented')

        # If the current clusters contain the same elements as they did last
        # time, we've converged
        raise Exception('kMeans: convergence not yet implemented')

    return protos

if __name__ == '__main__':
    insts = parseTrainingData()
    protos = kMeans(2, insts)
    for proto in protos: print proto
