class Instance:
    def __init__(self, data, label):
        '''Build an instance from a list of features and a list of target
        values'''
        self.data  = data   # The feature values - a list of numbers
        self.label = label  # The label/target value(s) - also a list of numbers

    def __str__(self):
        '''Get a nice string representation of this Instance object'''
        return ''.join(['DATA: ', str(self.data), ', LABEL: ', str(self.label)])
    
def parseTrainingData():
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

    # Break the float matrix into separate feature and target value matrices
    featureMatrix = []
    targetValMatrix = []
    for instData in floatMatrix:
        featureMatrix.append(instData[:len(instData) - 1])
        targetValMatrix.append([instData[len(instData) - 1]])

    # Convert the data matrix into a list of instances and return it
    return map(Instance, featureMatrix, targetValMatrix)

def ensureSameDimensionality(valueMatrix):
    '''Guarantee that, for a list of lists, all lists have the same length'''
    dimensionality = len(valueMatrix[0])
    for row in valueMatrix:
        if len(row) != dimensionality:
            raise Exception('Parsed data has varying dimensionality')
