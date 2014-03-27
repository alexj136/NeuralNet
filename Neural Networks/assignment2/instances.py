class Instance:
    def __init__(self, vec):
        '''Build an instance from a list of values where the last value is the
           class label'''
        self.vec = vec
        self.vlen = len(vec)    # compute the length now so that we don't have
                                # to recalculate it for every access
    def __str__(self):
        '''Get a nice string representation of this Instance object'''
        return ''.join(['LABEL: ', str(self.label), ', DATA: ', str(self.data)])
    @property
    def data(self):
        '''Get the data for this instance, without the label'''
        return self.vec[0:self.vlen-1]
    @property
    def label(self):
        '''Get the label for this instance'''
        return self.vec[self.vlen-1]

def parseInstances():
    '''Parse the file training_instances.txt into a list of Instance objects'''
    # Get the file as a string
    with open('training_data.txt', 'r') as f: text = f.read()

    # Split the string into a list of lines
    lines = text.split('\n')

    # Split each line into a list of strings, removing any empty lines
    splitLines = filter(lambda ln: ln != [], map(lambda s: s.split(), lines))

    # Parse a float from each string
    floatMatrix = map(lambda ln: map(float, ln), splitLines)

    # Ensure that all the instances have the same dimensionality
    ensureSameDimensionality(floatMatrix)

    # Convert the data matrix into a list of instances and return it
    return map(Instance, floatMatrix)

def ensureSameDimensionality(valueMatrix):
    '''Guarantee that, for a list of lists, all lists have the same length'''
    dimensionality = len(valueMatrix[0])
    for row in valueMatrix:
        if len(row) != dimensionality:
            raise Exception('Parsed data has varying dimensionality')
