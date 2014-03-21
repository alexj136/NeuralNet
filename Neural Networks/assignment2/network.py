class Network:
    def __init__(self, layers):
        '''Build a new Network from a list of Layers'''
        self.layers = layers
    def __str__(self):
        '''Get a nice string representation of this Network object'''
        return ''.join([''.join(["--- LAYER ", str(x), " ---\n",
            str(self.layers[x]), "\n"]) for x in range(len(self.layers))])

class Layer:
    def __init__(self, nodes):
        '''Build a new Layer from a list of nodes'''
        self.nodes = nodes
    def __str__(self):
        '''Get a nice string representation of this Layer object'''
        return ''.join([''.join([str(nd), "\n"]) for nd in self.nodes])

class Node:
    def __init__(self, wts):
        '''Build a new Node from a weight vector'''
        self.wts = wts
    def __str__(self):
        '''Get a nice string representation of this Node object'''
        return ''.join(['B: ', str(self.wts[0]), ', WTS: ', str(self.wts[1:])])

if __name__ == '__main__':
    network = Network(
        [ Layer(
            [ Node([0, 1])
            , Node([0, 1])
            ])
        , Layer(
            [ Node([0, 1])
            , Node([0, 1])
            ])
        ])
    print network
