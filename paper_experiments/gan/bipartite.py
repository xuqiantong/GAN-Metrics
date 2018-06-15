import numpy as np
import ot
from metric import distance

def solve(fake_feature,true_feature):
    # get the optimal matching between fake and true. assume #fake < # true

    M=distance(fake_feature,true_feature,True)
    emd = ot.emd([], [], M.numpy())

    map= np.zeros(fake_feature.size(0))

    for i in range(0,fake_feature.size(0)):
        for j in range(0,true_feature.size(0)):
            if emd[i][j]>0:
                map[i]=j
    return map

