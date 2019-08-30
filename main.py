from instances  import *
from mlp        import *
from rbf        import *
from preprocess import *
from misc       import *

def testNetEuclidean(net, testInsts):
    '''Test a network with the given test instances. Return the mean
    euclidean distance of the target output from the actual output.'''
    return sum([euclideanDist(net.fwdPass(inst), inst.label)
            for inst in testInsts])/len(testInsts)

def testNetMeanSquared(net, testInsts, ppr):
    '''Test a network with the given test instances. Return the mean
    squared error. Requires a Preprocessor object to rescale values.'''

    if isinstance(ppr, Preprocessor):
        factor = ppr.scaleInst.label[0]
    else:
        factor = ppr.preproc.scaleInst.label[0]

    return sum([sum([math.pow((t * factor) - (l * factor), 2)
            for t, l in zip(net.fwdPass(inst), inst.label)])
            for inst in testInsts])/len(testInsts)

def crossVal(net, numBins, insts, rate, convergenceThreshold, maxIters,
        usePCA=False):
    '''Perform cross-validation of the given network, with the given number
    of bins, with the given instances, with the given learning rate, with the
    given convergence threshold, and the given maximum number of iterations.
    Works for RBFNetworks and MLPNetworks.'''

    assert numBins > 1 # Can't cross-validate with less than two bins

    sets = bins(numBins, insts)
    eucTrainErrors = [] # Euclidean distance training errors
    eucGenErrors   = [] # Euclidean distance generalisation errors
    msqTrainErrors = [] # Mean squared training errors
    msqGenErrors   = [] # Mean squared generalisation errors

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

        eucTrainErrors.append(testNetEuclidean(net, pprTrainInsts))
        eucGenErrors.append(testNetEuclidean(net, pprTestInsts))
        msqTrainErrors.append(testNetMeanSquared(net, pprTrainInsts, ppr))
        msqGenErrors.append(testNetMeanSquared(net, pprTestInsts, ppr))

    eucTrainErr = sum(eucTrainErrors)/len(eucTrainErrors)
    eucGenErr   = sum(eucGenErrors)/len(eucGenErrors)
    msqTrainErr = sum(msqTrainErrors)/len(msqTrainErrors)
    msqGenErr   = sum(msqGenErrors)/len(msqGenErrors)

    if usePCA:
        return eucTrainErr * ppr.preproc.scaleInst.label[0], \
                eucGenErr * ppr.preproc.scaleInst.label[0], \
                msqTrainErr, msqGenErr
    else:
        return eucTrainErr * ppr.scaleInst.label[0], \
                eucGenErr * ppr.scaleInst.label[0], \
                msqTrainErr, msqGenErr

if __name__ == '__main__':

    # Perform tests of this implementation  - various MLP layouts & RBF
    # configurations. Should run in about 2 hours.

    insts = parseTrainingData()
    predInsts = parsePredictionData()

    # Test of chosen model on prediction data

    mlp = MLPNetwork(0, 0.3, [13, 13, 4, 1])
    pcaTrainInsts, ppr = pcaPreprocess(insts)
    pcaTestInsts = pcaPprWith(predInsts, ppr)
    mlp.train(pcaTrainInsts, 0.3, None, 1000)
    for inst, pcaInst in zip(predInsts, pcaTestInsts):
        print 'INST:', inst.data
        print 'PRED:', (mlp.fwdPass(pcaInst)[0] * \
                ppr.preproc.scaleInst.label[0]) + ppr.preproc.meanInst.label[0]

    # Cross-validation tests of various configurations

    for protos, withPCA in [(1, True), (5, True), (10, True), (20, True), \
            (1, False), (5, False), (10, False), (20, False)]:

        rbf = RBFNetwork(0, 0.3, protos, 1)
        eucTrainErr, eucGenErr, msqTrainErr, msqGenErr = \
                crossVal(rbf, 10, insts, 0.3, 0.001, 100, usePCA=withPCA)
        print 'RBF: PROTOS:', str(protos), 'PCA:', str(withPCA)
        print 'Euclidean Training error:', eucTrainErr
        print 'Euclidean Generalisation error:', eucGenErr
        print 'Mean Squared Training error:', msqTrainErr
        print 'Mean Squared Generalisation error:', msqGenErr

    for lyt, withPCA in [([13,         1], True ), ([13, 4,      1], True ), \
                         ([13, 13,     1], True ), ([13, 13, 4,  1], True ), \
                         ([13, 13, 13, 1], True ), ([13,         1], False), \
                         ([13, 4,      1], False), ([13, 13,     1], False), \
                         ([13, 13, 4,  1], False), ([13, 13, 13, 1], False)]:

        mlp = MLPNetwork(0, 0.3, lyt)
        eucTrainErr, eucGenErr, msqTrainErr, msqGenErr = \
                crossVal(mlp, 10, insts, 0.3, None, 1000, usePCA=withPCA)
        print 'MLP: LAYOUT:', str(lyt), 'PCA:', str(withPCA)
        print 'Euclidean Training error:', eucTrainErr
        print 'Euclidean Generalisation error:', eucGenErr
        print 'Mean Squared Training error:', msqTrainErr
        print 'Mean Squared Generalisation error:', msqGenErr
