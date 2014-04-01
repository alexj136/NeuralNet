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

def crossVal(n, insts, net):
    '''Perform cross-validation of this implementation on the given instances
    with a given n = number of bins. A good measure of generalisation error.'''

    sets = bins(10, insts)
    avgErrors = []

    for setIndex in range(n):
        testInsts  = sets[setIndex]
        trainInsts = flatten(sets[:setIndex] + sets[setIndex + 1:])

        net.resetGauss()
        net.trainBackProp(trainInsts, 0.05, 300)

        errors = [euclideanDist(net.fwdPass(inst), inst.label)
                for inst in testInsts]

        avgErrors.append(sum(errors)/len(errors))

    return sum(avgErrors)/len(avgErrors)

if __name__ == '__main__':

    insts = parseTrainingData()
    dmdInsts, meanInst = demean(insts)
    scldInsts, scaleInst = scale(dmdInsts)

    net = Network.gaussWtsNet(0, 0.3, 13, [13, 5, 1])

    err = crossVal(10, insts, net)

    print str((err * scaleInst.label[0]) + meanInst.label[0])
