from instances  import *
from network    import *
from preprocess import *

if __name__ == '__main__':
    insts = parseInstances()
    net   = Network.gaussWtsNet(0, 0.3, 13, [13, 13, 2])
    print net
    print str(insts[0].label)
    print str(net.fwdPass(insts[0]))
    means = demean(insts)
    sfs   = scale(insts)
#    for inst in insts:
#        print inst
    print '----- means -----'
    print means
    print '----- scales -----'
    print sfs
