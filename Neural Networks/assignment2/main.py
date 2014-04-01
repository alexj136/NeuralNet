from instances  import *
from network    import *
from preprocess import *

def crossVal(bins, insts, net):
    '''Perform cross-validation of this implementation on the given instances
    with a given number of bins. A good measure of generalisation error.'''

    # Break the instance list up into a list of n lists where n = bins
    sets = []
    instIndex = 0
    for x in range(bins):
        sets.insert(0, [])
        for y in range(len(insts)/bins):
            if instIndex >= len(insts): break
            sets[0].append(insts[instIndex])
            instIndex = instIndex + 1
        if instIndex >= len(insts): break

    setIndex = 0
    avgErrors = []
    while setIndex < bins:
        testInsts  = sets[setIndex]
        trainInstBins = sets[:setIndex] + sets[setIndex + 1:]
        trainInsts = []
        for instBin in trainInstBins:
            for inst in instBin:
                trainInsts.append(inst)

        net.resetGauss()
        net.trainBackProp(trainInsts, 0.1, 100)

        errors = [euclideanDist(net.fwdPass(inst), inst.label)
                for inst in testInsts]

        avgErrors.append(sum(errors)/len(errors))

        setIndex = setIndex + 1

    return sum(avgErrors)/len(avgErrors)

if __name__ == '__main__':

    insts     = parseInstances()
    meanInst  = demean(insts)
    scaleInst = scale(insts)

    net = Network.gaussWtsNet(0, 0.3, 13, [13, 13, 1])

    err = crossVal(10, insts, net)

    print str((err * scaleInst.label[0]) + meanInst.label[0])
