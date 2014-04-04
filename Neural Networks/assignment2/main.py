from instances  import *
from network    import *
from preprocess import *

def bins(n, data):
    '''Break a single list up into a list of n lists, differing in size by no
    more than 1. Does not modify the original list, but data is not copied.
    Ordering of elements in output is arbitrary.'''
    sets = [[] for x in range(n)]

    setIndex = 0
    elemIndex = 0

    while elemIndex < len(data):
        sets[setIndex].append(data[elemIndex])
        elemIndex = elemIndex + 1
        setIndex = setIndex + 1 if setIndex + 1 < len(sets) else 0

    return sets

def flatten(lists):
    '''Flatten a list of lists into one list, maintaining ordering. As with
    bins(), the given list is not modified, but data is not copied.'''
    return [elem for lst in lists for elem in lst]

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

xorData = \
        [ Instance([-1, -1], [-1])
        , Instance([-1,  1], [ 1])
        , Instance([ 1, -1], [ 1])
        , Instance([ 1,  1], [-1])
        ]
if __name__ == '__main__':

    insts = parseTrainingData()
    trainErr, genErr = crossVal(4, 0, 3, [13, 5, 5, 5, 1], insts, 0.5, 1000)
    print 'Training error:', trainErr
    print 'Generalisation error:', genErr
