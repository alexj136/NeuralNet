import matplotlib.pyplot as plt
from kmeans import kMeans
from main import *

insts = parseTrainingData()

xs = [inst.data[5] for inst in insts]
ys = [inst.data[7] for inst in insts]

insts = [Instance([x, y], []) for x, y in zip(xs, ys)]

protos = kMeans(20, insts)
pxs = [proto.data[0] for proto in protos]
pys = [proto.data[1] for proto in protos]

plt.plot(xs, ys, 'bo', pxs, pys, 'r^')
plt.show()
