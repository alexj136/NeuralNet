def demean(insts):
    '''Demean the set of instances. This amounts to adding or subtracting a
    fixed value from each value for each instance such that the mean of each
    feature is zero. Returns a list of the original mean values of each feature
    so that the same transformation can be applied to unseen instances.'''

    # We record the original mean of each feature and return them in a list, so
    # that we can (a) apply an identical transformation to unseen instances and
    # (b) recover the feature values as they were before demeaning
    means = []

    for feat in range(len(insts[0].vec)):

        # Compute the sum total of all values for this feature in order to find
        # the mean by dividing by the number of instances
        total = 0
        for inst in insts:
            total = total + inst.vec[feat]

        # Calculate the mean
        mean = total / len(insts)

        # Record this mean
        means.append(mean)

        # Subtract the mean from each value
        for inst in insts:
            inst.vec[feat] = inst.vec[feat] - mean
    
    return means

def demeanNewInst(inst, means):
    '''Given an instance and a list of mean values for data returned by
    demean(), subtract each mean from its corresponding feature in the given
    instance.'''
    for feat in range(len(inst.vec)):
        inst.vec[feat] = inst.vec[feat] - means[feat]

def scale(insts):
    '''Scale a set of instances. This amounts to dividing the value of each
    feature by the absolute value of the maximum value for that feature in the
    data set. As with demean, we return a list of scale factors used so that we
    can apply the same scaling to unseen instances.'''

    # We record the scale factor for feature and return them in a list, so
    # that we can (a) apply an identical transformation to unseen instances and
    # (b) recover the feature values as they were before scaling
    scaleFacs = []

    for feat in range(len(insts[0].vec)):

        # Find the maximum value for this feature, which we will use as our
        # scale factor
        absMax = 0
        for inst in insts:
            if abs(inst.vec[feat]) > absMax:
                absMax = abs(inst.vec[feat])

        # Record this scale factor
        scaleFacs.append(absMax)

        # Divide each feature value by the scale factor
        for inst in insts:
            inst.vec[feat] = inst.vec[feat] / absMax

    return scaleFacs

def scaleNewInst(inst, scaleFacs):
    '''Given an instance and a list of scale factors for data returned by
    scale(), divide each feature value by the corresponding scale factor.'''
    for feat in range(len(inst.vec)):
        inst.vec[feat] = inst.vec[feat] / scaleFacs[feat]
