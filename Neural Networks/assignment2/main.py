from instances  import *
from network    import *
from preprocess import *

if __name__ == '__main__':

    insts = parseInstances()
    means = demean(insts)
    sfs   = scale(insts)

    net   = Network.gaussWtsNet(0, 0.3, 13, [4, 1])
    print str(insts[0].label)
    fwp = net.fwdPass(insts[0])
    for layer in fwp:
        print str(layer)
