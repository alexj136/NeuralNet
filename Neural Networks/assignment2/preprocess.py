from instances import Instance

def demean(insts):
    '''Demean the set of instances. This amounts to adding or subtracting a
    fixed value from each value for each instance such that the mean of each
    feature is zero. Returns a list of the original mean values of each feature
    so that the same transformation can be applied to unseen instances.'''

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

        # Subtract the mean from each value
        for inst in insts:
            inst.data[feat] = inst.data[feat] - mean

    targetValMeans = []    

    for targetDim in range(len(insts[0].label)):

        # Compute the sum total of all values for this dimension of the target
        # in order to find the mean by dividing by the number of instances
        total = 0
        for inst in insts:
            total = total + inst.label[targetDim]

        # Calculate the mean
        mean = total / len(insts)

        # Record this mean
        targetValMeans.append(mean)

        # Subtract the mean from each value
        for inst in insts:
            inst.label[targetDim] = inst.label[targetDim] - mean
    
    return Instance(featMeans, targetValMeans)

def demeanNewInst(inst, means):
    '''Given an instance and a set of mean values for data returned by
    demean() stored in an Instance obhect, subtract each mean from its
    corresponding feature/target value in the given instance.'''
    for feat in range(len(inst.data)):
        inst.data[feat] = inst.data[feat] - means.data[feat]
    for targetDim in range(len(inst.label)):
        inst.label[targetDim] = inst.label[targetDim] - means.label[targetDim]

def scale(insts):
    '''Scale a set of instances. This amounts to dividing the value of each
    feature by the absolute value of the maximum value for that feature in the
    data set. As with demean, we return a list of scale factors used so that we
    can apply the same scaling to unseen instances.'''

    # We record the scale factor for feature and return them in a list, so
    # that we can (a) apply an identical transformation to unseen instances and
    # (b) recover the feature values as they were before scaling
    featScaleFacs = []

    for feat in range(len(insts[0].data)):

        # Find the maximum value for this feature, which we will use as our
        # scale factor
        absMax = 0
        for inst in insts:
            if abs(inst.data[feat]) > absMax:
                absMax = abs(inst.data[feat])

        # Record this scale factor
        featScaleFacs.append(absMax)

        # Divide each feature value by the scale factor
        for inst in insts:
            inst.data[feat] = inst.data[feat] / absMax

    targetValScaleFacs = []    

    for targetDim in range(len(insts[0].label)):

        # Find the maximum value for this target dimension, which we will use as
        # our scale factor
        absMax = 0
        for inst in insts:
            if abs(inst.label[targetDim]) > absMax:
                absMax = abs(inst.label[targetDim])

        # Record this scale factor
        targetValScaleFacs.append(absMax)

        # Divide each target value by the scale factor
        for inst in insts:
            inst.label[targetDim] = inst.label[targetDim] / absMax

    return Instance(featScaleFacs, targetValScaleFacs)

def scaleNewInst(inst, scaleFacs):
    '''Given an instance and a set of scale factors for data returned by
    scale() stored in an Instance object, divide each feature value by the
    corresponding scale factor.'''
    for feat in range(len(inst.data)):
        inst.data[feat] = inst.data[feat] / scaleFacs.data[feat]
    for targetDim in range(len(inst.label)):
        inst.label[targetDim] = \
                inst.label[targetDim] / scaleFacs.label[targetDim]
