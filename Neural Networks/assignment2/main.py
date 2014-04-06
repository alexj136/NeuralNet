from instances  import *
from mlp        import *
from rbf        import *
from preprocess import *
from misc       import *

def trainNet(mean, stdDev, layout, trainInsts, rate, iters):
    '''Train a network with the given training instances, initialised with
    random weights drawn from a gaussian distribution with the given mean and
    standard deviation. The Network will have the given layout (see the
    docstring for Network.gaussWtsNet()), and will be trained with the given
    learning rate, for the given number of iterations.'''

    net = Network.gaussWtsNet(mean, stdDev, layout)
    net.trainBackProp(trainInsts, rate, iters)

    return net

def testNet(net, testInsts):
    '''Test this network with the given test instances. Return the mean
    euclidean distance of the target output from the actual output.'''
    return sum([euclideanDist(net.fwdPass(inst), inst.label)
            for inst in testInsts])/len(testInsts)

def crossVal(numBins, mean, stdDev, layout, insts, rate, iters):

    sets = bins(numBins, insts)
    trainErrors = [] # Training errors
    genErrors   = [] # Generalisation errors

    for setIndex in range(numBins):
        testInsts  = sets[setIndex]
        trainInsts = flatten(sets[:setIndex] + sets[setIndex + 1:])

        preprocTestInsts, preproc = preprocess(trainInsts)
        preprocTrainInsts = preprocessWith(testInsts, preproc)

        net = trainNet(mean, stdDev, layout, trainInsts, rate, iters)

        trainErrors.append(testNet(net, preprocTrainInsts))
        genErrors.append(testNet(net, preprocTestInsts))

    avTrainErr = sum(trainErrors)/len(trainErrors)
    avGenErr   = sum(genErrors)/len(genErrors)

    return (avTrainErr * preproc.scaleInst.label[0]) + \
            preproc.meanInst.label[0], \
            (avGenErr * preproc.scaleInst.label[0]) + preproc.meanInst.label[0]

if __name__ == '__main__':

    insts = parseTrainingData()
    #trainErr, genErr = crossVal(4, 0, 3, [13, 5, 5, 5, 1], insts, 0.5, 1000)
    #print 'Training error:', trainErr
    #print 'Generalisation error:', genErr

    net = RBFNetwork(5, 1)
    tr = insts[:400]
    te = insts[400:]
    net.train(0, 0.3, tr, 0.1, 100)
    for inst in te:
        print 'LABEL:', inst.label, 'PRED:', net.fwdPass(inst)
