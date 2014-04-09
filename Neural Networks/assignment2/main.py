from instances  import *
from mlp        import *
from rbf        import *
from preprocess import *
from misc       import *

def testNet(net, testInsts):
    '''Test a network with the given test instances. Return the mean
    euclidean distance of the target output from the actual output.'''
    return sum([euclideanDist(net.fwdPass(inst), inst.label)
            for inst in testInsts])/len(testInsts)

def crossVal(net, numBins, insts, rate, convergenceThreshold, maxIters,
        usePCA=False):
    '''Perform cross-validation of the given network, with the given number
    of bins, with the given instances, with the given learning rate, with the
    given convergence threshold, and the given maximum number of iterations.
    Works for RBFNetworks and MLPNetworks.'''

    assert numBins > 1 # Can't cross-validate with less than two bins

    sets = bins(numBins, insts)
    trainErrors = [] # Training errors
    genErrors   = [] # Generalisation errors

    for setIndex in range(numBins):
        testInsts  = sets[setIndex]
        trainInsts = flatten(sets[:setIndex] + sets[setIndex + 1:])

        if usePCA:
            pprTestInsts, ppr = pcaPreprocess(trainInsts)
            pprTrainInsts = pcaPprWith(testInsts, ppr)
        else:
            pprTestInsts, ppr = preprocess(trainInsts)
            pprTrainInsts = pprWith(testInsts, ppr)

        net.reinitialise()
        net.train(pprTrainInsts, rate, convergenceThreshold, maxIters)

        trainErrors.append(testNet(net, pprTrainInsts))
        genErrors.append(testNet(net, pprTestInsts))

    meanTrainErr = sum(trainErrors)/len(trainErrors)
    meanGenErr   = sum(genErrors)/len(genErrors)

    if usePCA:
        for i, ppi in zip(sets[0], pprTestInsts):
            print 'LBL:', i.label, 'PRD:', \
                    unppr(Instance([], net.fwdPass(ppi)), ppr.preproc).label
        return meanTrainErr * ppr.preproc.scaleInst.label[0], \
                meanGenErr * ppr.preproc.scaleInst.label[0]
    else:
        for i, ppi in zip(sets[0], pprTestInsts):
            print 'LBL:', i.label, 'PRD:', \
                    unppr(Instance([], net.fwdPass(ppi)), ppr).label

        return meanTrainErr * ppr.scaleInst.label[0], \
                meanGenErr * ppr.scaleInst.label[0]

if __name__ == '__main__':

    insts = parseTrainingData()

    '''
    rbf = RBFNetwork(0, 0.3, 20, 1)
    trainErr, genErr = crossVal(rbf, 2, insts, 0.3, 0.001, 100, usePCA=True)
    #trainErr, genErr = crossVal(rbf, 2, insts, 0.3, 0.001, 100, usePCA=False)
    print 'Training error:', trainErr
    print 'Generalisation error:', genErr

    '''

    mlp = MLPNetwork(0, 0.3, [13, 13, 1])
    #trainErr, genErr = crossVal(mlp, 2, insts, 0.3, None, 100, usePCA=True)
    trainErr, genErr = crossVal(mlp, 2, insts, 0.3, None, 100, usePCA=False)
    print 'Training error:', trainErr
    print 'Generalisation error:', genErr

