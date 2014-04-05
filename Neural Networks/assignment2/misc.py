from instances import Instance

xorData = \
        [ Instance([-1, -1], [-1])
        , Instance([-1,  1], [ 1])
        , Instance([ 1, -1], [ 1])
        , Instance([ 1,  1], [-1])
        ]

def euclideanDist(x, y):
    '''Compute the euclidean distance between two vectors (lists) x and y'''
    return math.sqrt(sum(map(lambda xi, yi: math.pow(xi - yi, 2), x, y)))

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
