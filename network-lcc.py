# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import numpy.ma as ma
import networkx as nx
from matplotlib.pyplot import plot


# ========================= Distance ========================= #
def Distance(SD, modFrom, modTo=None):
    if modTo is None:
        return SD[np.ix_(modFrom, modFrom)].min(1).mean()
    else:
        return SD[np.ix_(modFrom, modTo)].min(1).mean()


def Shortest(SD, mod1, mod2):
    return SD[np.ix_(mod1, mod2)].mean()


def Closest(SD, mod1, mod2):
    sub = SD[np.ix_(mod1, mod2)]
    closest1 = sub.min(0)
    closest2 = sub.min(1)
    return (closest1.sum() + closest2.sum()) / (closest1.count() + closest2.count())


def Separation(SD, mod1, mod2):
    dAA = Distance(SD, mod1)
    dBB = Distance(SD, mod2)
    dAB = Closest(SD, mod1, mod2)
    return dAB - (dAA + dBB) / 2


def Center(SD, mod1, mod2):
    c1 = GetCenterNodes(SD, mod1)
    c2 = GetCenterNodes(SD, mod2)
    return SD[np.ix_(c1, c2)].mean()


def GetCenterNodes(SD, mod):
    sum_ = SD[np.ix_(mod, mod)].sum(1)
    return [mod[idx] for idx, isCenter in enumerate(sum_ == sum_.min()) if isCenter]  # bool(numpy.ma.core.MaskedConstant) => False


def Kernel(SD, mod1, mod2):
    sub = SD[np.ix_(mod1, mod2)]
    exp = ma.exp(-sub)
    alb = ma.log(exp.sum(1))
    bla = ma.log(exp.sum(0))
    nA = alb.count()
    nB = bla.count()
    return (nA * np.log(nB) + nB * np.log(nA) - alb.sum() - bla.sum()) / (nA + nB) + 1


# ========================= Network ========================== #
class Network(object):
    DISTANCE_MEASURE = {"DISTANCE"  : Distance,
                        "SHORTEST"  : Shortest,
                        "CLOSEST"   : Closest,
                        "SEPARATION": Separation,
                        "CENTER"    : Center,
                        "KERNEL"    : Kernel}

    def __init__(self, pathG, pathSD):
        self.G = nx.read_adjlist(pathG)
        self.SD = np.load(pathSD)
        self.nodes = sorted(self.G.nodes())
        self.i2n = {index: node for index, node in enumerate(self.nodes)}
        self.n2i = {node: index for index, node in enumerate(self.nodes)}
        self.i2d = {}  # index: degree
        self.d2i = {}  # degree: [index1, index2, ...]
        for node, degree in self.G.degree():
            index = self.n2i[node]
            if degree in self.d2i:
                self.d2i[degree].append(index)
            else:
                self.d2i[degree] = [index]
            self.i2d[index] = degree
        self.dmin = min(self.d2i.keys())
        self.dmax = max(self.d2i.keys())
        self.d2b = {}  # degree: bin
        self.b2i = {}  # bin: [index1, index2, ...]

    def Name2Index(self, names, skipUnknown=True):
        if skipUnknown:
            return [self.n2i[n] for n in names if n in self.n2i]
        else:
            return [self.n2i[n] for n in names]

    def Index2Name(self, indexes, skipUnknown=True):
        if skipUnknown:
            return [self.i2n[i] for i in indexes if i in self.i2n]
        else:
            return [self.i2n[i] for i in indexes]

    # ------------------------------------------
    def PrepareBins(self, binSize=100):
        index = 0
        self.d2b = {}
        self.b2i[index] = []
        degrees = sorted(self.d2i.keys())
        for curr, next in zip(degrees, degrees[1:] + [degrees[-1] + 1]):
            for d in range(curr, next):
                self.d2b[d] = index
            self.b2i[index].extend(self.d2i[curr])
            if curr != degrees[-1] and len(self.b2i[index]) >= binSize:
                index += 1
                self.b2i[index] = []
        if len(self.b2i[index]) < binSize and index > 0:  # Merge last two bins if last bin < binSize
            for d in range(degrees[-1], -1, -1):
                if self.d2b[d] != index:
                    break
                self.d2b[d] = index - 1
            self.b2i[index - 1].extend(self.b2i[index])
            del self.b2i[index]

    def DegreeMimicSampling(self, indexes):
        binCount = {}
        for index in indexes:
            d = self.i2d[index]
            if d < self.dmin:
                d = self.dmin
            elif d > self.dmax:
                d = self.dmax
            b = self.d2b[d]
            if b in binCount:
                binCount[b] += 1
            else:
                binCount[b] = 1
        while True:  # TODO ValueError
            yield sum([random.sample(self.b2i[b], binCount[b]) for b in sorted(binCount.keys())], [])

    def TotalRandomSampling(self, n):
        indexes = list(range(len(self.nodes)))
        while True:
            yield random.sample(indexes, n)

    # ------------------------------------------
    def Proximity(self, mod1, mod2, method="DISTANCE"):
        return self.DISTANCE_MEASURE[method](self.SD, mod1, mod2)

    def ProximityRandom(self, mod1, mod2, method="DISTANCE", repeat=1000, seed=None):
        if seed is not None:
            random.seed(seed)
        method = self.DISTANCE_MEASURE[method]
        result = np.zeros(repeat)
        index = 0
        for mod1r, mod2r in zip(self.DegreeMimicSampling(mod1), self.DegreeMimicSampling(mod2)):
            v = method(self.SD, mod1r, mod2r)
            if not ma.is_masked(v):
                result[index] = v
                index += 1
                if index == repeat:
                    break
        return result

    def LCC(self, mod):
        return len(max(nx.connected_components(self.G.subgraph(self.Index2Name(mod))), key=len))

    def LCCRandom(self, mod, repeat=1000, seed=None):
        if seed is not None:
            random.seed(seed)
        result = np.zeros(repeat)
        index = 0
        for modr in self.DegreeMimicSampling(mod):
            result[index] = self.LCC(modr)
            index += 1
            if index == repeat:
                break
        return result


# ========================= Utility ========================== #
def Tsv2Adj(pathIn, pathOut, encoding="utf-8", removeSelfloop=True):
    G = nx.Graph()
    with open(pathIn, encoding=encoding) as fi:
        for line in fi:
            if not line.startswith("#"):
                G.add_edge(*line.strip().split("\t")[0:2])
    if removeSelfloop:
        G.remove_edges_from(nx.selfloop_edges(G))
    nx.write_adjlist(G, pathOut)
    return G


def Z_Score(real, background):
    m = background.mean()
    s = background.std(ddof=1)
    z = (real - m) / s
    p = np.mean(background < real)  # left
    # print(real, background.min(), background.max())
    # print(m, s, z, p)
    return z, p


# ========================= ======== ========================= #
if __name__ == "__main__":
    import matplotlib.pyplot as plot

    files = [
        "AD_proteomics_freq2.txt",
        "AD_proteomics_freq3.txt",
        "AD_proteomics_freq4.txt",
        "AD_proteomics_freq5.txt",
        "AD_proteomics_freq6.txt",
        "vascular_dysfunction_27.txt",
        "AD_seed_Nature_Aging144.txt",
        "Amyloid_Nature_Aging54.txt",
        "Lysosomal_dysfunction17.txt",
        "Mitochondrial_dysfunction18.txt",
        "Neuroinflammation38.txt",
        "Tau_Nature_Aging27.txt",
          ]

    Net = Network("HumanInteractome2022.adj", "HumanInteractome2022.npy")
    Net.PrepareBins(100)

    for file1 in files:
        G1 = Net.Name2Index(set(open(os.path.join("input", file1)).read().splitlines()))
        print(len(G1))

        d = Net.LCC(G1)
        for p in nx.connected_components(Net.G.subgraph(Net.Index2Name(G1))):
            if len(p)==d:
                print(p)


        b = Net.LCCRandom(G1, repeat=10000, seed=1024)

        plot.figure()
        plot.hist(b, bins=50)
        plot.savefig(os.path.join("output", file1) + ".png")

        plot.figure()
        plot.hist([Net.i2d[i] for i in G1])
        plot.savefig(os.path.join("output", file1) + "-degree.png")

        plot.figure()
        for xxx in Net.DegreeMimicSampling(G1):
            plot.hist([Net.i2d[i] for i in xxx])
            break
        plot.savefig(os.path.join("output", file1) + "-degree-random.png")

        with open(os.path.join("output", file1), "w") as fo:
            for v in b:
                fo.write("%.6f\n" % v)
        print(file1,d, Z_Score(d, b))
