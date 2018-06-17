import math
import os
# from gurobipy import *
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as models
import pdb
from tqdm import tqdm

from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1).cuda()
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1).cuda()
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX,nY) + Y2.expand(nY,nX).transpose(0,1) - 2*torch.mm(X,Y.transpose(0,1)))

    del X, X2, Y, Y2
    
    if sqrt:
        M = ((M+M.abs())/2).sqrt()
    
    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([],[],M.numpy())

    return emd



class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0),torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx,Mxy),1), torch.cat((Mxy.transpose(0,1),Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M+torch.diag(INFINITY*torch.ones(n0+n1))).topk(k, 0, False)

    count = torch.zeros(n0+n1)
    for i in range(0,k):
        count = count + label.index_select(0,idx[i])
    pred = torch.ge(count, (float(k)/2)*torch.ones(n0+n1)).float()

    s = Score_knn()
    s.tp = (pred*label).sum()
    s.fp = (pred*(1-label)).sum()
    s.fn = ((1-pred)*label).sum()
    s.tn = ((1-pred)*(1-label)).sum()
    s.precision = s.tp/(s.tp+s.fp)
    s.recall = s.tp/(s.tp+s.fn)
    s.acc_t = s.tp/(s.tp+s.fn)
    s.acc_f = s.tn/(s.tn+s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k 

    return s


def mmd(Mxx, Mxy, Myy, sigma) :
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = torch.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = torch.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()
    mmd = math.sqrt(max(a, 0))

    return mmd


def ent(M, epsilon) :
    n0 = M.size(0)
    n1 = M.size(1)
    neighbors = M.lt(epsilon).float()
    sums = neighbors.sum(0).repeat(n0,1)
    sums[sums.eq(0)] = 1
    neighbors = neighbors.div(sums)
    probs = neighbors.sum(1) / n1
    rem = 1 - probs.sum()
    if rem < 0:
        rem = 0
    probs = torch.cat((probs,rem*torch.ones(1)),0)
    e = {}
    e['probs'] = probs
    probs = probs[probs.gt(0)]
    e['ent'] = -probs.mul(probs.log()).sum()

    return e


def entropy_score(X, Y, epsilons):
    Mxy = distance(X, Y, False)
    scores = []
    for epsilon in epsilons:
        scores.append(ent(Mxy.t(), epsilon))
    
    return scores


eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score


def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())
    
    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2*m_w.dot(m) + np.trace(C + C_w - 2*C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    emd = 0
    mmd = 0
    knn = None
    def printScore(self):
        print("-------------------------")
        print("\t\tEMD:",self.emd)
        print("\t\tMMD:",self.mmd)
        print("\t\tPrecision:",self.knn.precision)
        print("\t\tRecall:",self.knn.recall)
        print("\t\tAcc_t:",self.knn.acc_t)
        print("\t\tAcc_f:",self.knn.acc_f)
        print("-------------------------")

    def printScore(self):
        print("-------------------------")
        print("\t\tEMD:",self.emd)
        print("\t\tMMD:",self.mmd)
        print("\t\tPrecision:",self.knn.precision)
        print("\t\tRecall:",self.knn.recall)
        print("\t\tAcc_t:",self.knn.acc_t)
        print("\t\tAcc_f:",self.knn.acc_f)
        print("-------------------------")


def compute_score(real, fake, k=1, sigma=1, sqrt=True):

    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    s.printScore()
    
    return s
