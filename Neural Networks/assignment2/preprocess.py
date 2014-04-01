from instances import Instance
from copy      import deepcopy

def demean(insts):
    '''Produce a new list of instances from the given ones, that have been
    demeaned. Return a new list, and an Instance object that represents the mean
    values of the data from the original instances, that can be used to demean
    unseen instances, or to recover values in the original domain'''

    # We record the original mean of each feature and return them in a list, so
    # that we can (a) apply an identical transformation to unseen instances and
    # (b) recover the feature values as they were before demeaning
    featMeans = []

    for feat in range(len(insts[0].data)):

        # Compute the sum total of all values for this feature in order to find
        # the mean by dividing by the number of instances
        total = 0
        for inst in insts:
            total = total + inst.data[feat]

        # Calculate the mean
        mean = total / len(insts)

        # Record this mean
        featMeans.append(mean)

    labelMeans = []    

    for lbl in range(len(insts[0].label)):

        # Compute the sum total of all values for this dimension of the target
        # in order to find the mean by dividing by the number of instances
        total = 0
        for inst in insts:
            total = total + inst.label[lbl]

        # Calculate the mean
        mean = total / len(insts)

        # Record this mean
        labelMeans.append(mean)

    # Create new instances from the old ones and adjust all values appropriately
    newInsts = deepcopy(insts)
    for inst in newInsts:
        for d in range(len(inst.data)):
            inst.data[d] = inst.data[d] - featMeans[d]
        for l in range(len(inst.label)):
            inst.label[l] = inst.label[l] - labelMeans[l]
    
    return newInsts, Instance(featMeans, labelMeans)

def demeanNewInst(inst, means):
    '''Given an unseen Instance and an example mean Instance, return a new
    instance in the same domain as the data that produced the example
    Instance'''
    newInst = deepcopy(inst)
    for feat in range(len(newInst.data)):
        newInst.data[feat] = newInst.data[feat] - means.data[feat]
    for lbl in range(len(newInst.label)):
        newInst.label[lbl] = newInst.label[lbl] - means.label[lbl]
    return newInst

def scale(insts):
    '''Produce a list of Instances that are scaled versions of the given
    Instances. Return the new instances and an example Instance that stores the
    scale factors for each dimension of the given instances.'''

    # We record the scale factor for feature and return them in a list, so
    # that we can (a) apply an identical transformation to unseen instances and
    # (b) recover the feature values as they were before scaling
    featScales = []

    for feat in range(len(insts[0].data)):

        # Find the maximum value for this feature, which we will use as our
        # scale factor
        absMax = 0
        for inst in insts:
            if abs(inst.data[feat]) > absMax:
                absMax = abs(inst.data[feat])

        # Record this scale factor
        featScales.append(absMax)

    labelScales = []    

    for lbl in range(len(insts[0].label)):

        # Find the maximum value for this target dimension, which we will use as
        # our scale factor
        absMax = 0
        for inst in insts:
            if abs(inst.label[lbl]) > absMax:
                absMax = abs(inst.label[lbl])

        # Record this scale factor
        labelScales.append(absMax)

    # Create new instances from the old ones and adjust all values appropriately
    newInsts = deepcopy(insts)
    for inst in newInsts:
        for d in range(len(inst.data)):
            inst.data[d] = inst.data[d] / featScales[d]
        for l in range(len(inst.label)):
            inst.label[l] = inst.label[l] / labelScales[l]

    return newInsts, Instance(featScales, labelScales)

def scaleNewInst(inst, scaleFacs):
    '''Given an instance and a scale factor example Instance, for data returned
    by scale(), return a new instance that is equivalent to the given instance,
    but in the scaled domain.'''
    newInst = deepcopy(inst)
    for feat in range(len(newInst.data)):
        newInst.data[feat] = newInst.data[feat] / scaleFacs.data[feat]
    for lbl in range(len(newInst.label)):
        newInst.label[lbl] = newInst.label[lbl] / scaleFacs.label[lbl]
    return newInst
