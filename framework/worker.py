import argparse

import numpy as np
from subprocess import call

import os
from globals import Globals, get_hostname
from sampler.noise import noise_sampler
from sampler.realData import folder_sampler
from sampler.generator import generator_sampler
from sampler.subclass import subclass_sampler
from utils import mkdir
from mix import getDat, Ent
from metric import distance, compute_score, inception_score, entropy_score, mode_score, fid
from experiments import run_overfit_exp, run_collapse_exp, run_drop_exp
from gan.DCGAN import DCGAN_main
from gan.DCGAN_cluster import DCGAN_cluster_main
from gan.MGGAN import MGGAN_main
from gan.NNGAN import NNGAN_main
from gan.WGAN import WGAN_main
from utils import pickOne, print_prop
import pickle
import scipy.io as sio

g = Globals()
parser = argparse.ArgumentParser()
parser.add_argument('--A', default=1, type=int, help='minimum machine id?')
parser.add_argument('--B', default=20, type=int, help='maximum machine id?')
parser.add_argument('--id', default="M1", type=str, help='id')
parser.add_argument('--serial', action="store_true", help='hostname')
parser.add_argument('--f', type=str, default="tasks.npy", help='hostname')

opt = parser.parse_args()

labels = g.default_repo_dir + "labels/"
mkdir(g.default_repo_dir)
mkdir(labels)
mkdir(labels)
labels = labels + "/"

hostname = get_hostname()
if not hostname.startswith("M"):
    hostname = opt.id
tasks = np.load(g.default_repo_dir + opt.f)


def run_combine(t):
    print_prop(t)
    if t.mode == "Mix" or t.mode == "Entropy":
        mkdir(g.default_repo_dir + "samples/Mix")
        print("running mix mod..")
        mkdir(g.default_repo_dir + "rlt")
        for sp in range(0, 5 if t.mixOnly else 1):
            def getName():
                ans = 'mix_' if t.mode == "Mix" else 'entropy_'
                ans = ans + t.data + '_' + t.featureType + '_'
                for entry in t.dataList:
                    ans = ans + entry.data + '_' + entry.folder + '_' + \
                        ('%.2f' % entry.fraction) + '_' + \
                        str(entry.dup) + '_' + str(entry.imageMode) + '_'
                return ans + str(t.mixSize) + '_' + str(sp)
            saveFile = g.default_repo_dir + "rlt/" + getName()

            if os.path.exists(saveFile):
                print(saveFile + " already generated. Continue.")
                continue

            if t.mode == "Mix":
                mixDat = getDat(t, t.dataList, g.default_repo_dir +
                                "samples/Mix/M1" + hostname, t.featureType, t.mixOnly)
                dataList = [Ent(1, t.data, 'true', None)]
                print('ftype', t.featureType)
                trueDat = getDat(t, dataList, g.default_repo_dir +
                                 "samples/Mix/M2" + hostname, t.featureType)
                score = compute_score(trueDat, mixDat, 1, 1, True)
            else:
                t.mixSize = t.mixSize * 10
                mixDat = getDat(t, t.dataList, g.default_repo_dir +
                                "samples/Mix/M1" + hostname, t.featureType, t.mixOnly)
                t.mixSize = t.mixSize / 10
                dataList = [Ent(1, t.data, 'true_test', None)]
                trueDat = getDat(t, dataList, g.default_repo_dir +
                                 "samples/Mix/M2" + hostname, t.featureType)
                if t.featureType == 'pix':
                    epsilons = [1000]
                elif t.featureType == 'smax':
                    epsilons = [0.001]
                elif t.featureType == 'class':
                    epsilons = [0.1]
                elif t.featureType == 'conv':
                    epsilons = [1000]
                score = entropy_score(mixDat, trueDat, epsilons)
            f = open(saveFile, "wb")
            pickle.dump(score, f)
            f.close()

    elif t.mode == "Incep":
        mkdir(g.default_repo_dir + "samples/Mix")
        mkdir(g.default_repo_dir + "rlt")
        for sp in range(0, 5):
            def getIncepName():
                ans = 'incep_' + t.data + '_' + t.featureType + '_'
                for entry in t.dataList:
                    ans = ans + entry.data + '_' + entry.folder + '_' + \
                        ('%.2f' % entry.fraction) + '_' + \
                        str(entry.dup) + '_' + str(entry.imageMode) + '_'
                return ans + str(t.mixSize) + "_" + str(sp)
            incep_file_name = g.default_repo_dir + "rlt/" + getIncepName()
            if os.path.exists(incep_file_name):
                print(incep_file_name + " already generated. Continue.")
                return
            X = getDat(t, t.dataList, g.default_repo_dir + "samples/Mix/M1" +
                       hostname, t.featureType)  # get smax feature
            incep_score = inception_score(X)
            f = open(incep_file_name, "wb")
            pickle.dump(incep_score, f)
            f.close()

    elif t.mode == "ModeScore":
        mkdir(g.default_repo_dir + "samples/Mix")
        mkdir(g.default_repo_dir + "rlt")
        for sp in range(0, 5):
            def getMSName():
                ans = 'mode_score_' + t.data + '_' + t.featureType + '_'
                for entry in t.dataList:
                    ans = ans + entry.data + '_' + entry.folder + '_' + \
                        ('%.2f' % entry.fraction) + '_' + \
                        str(entry.dup) + '_' + str(entry.imageMode) + '_'
                return ans + str(t.mixSize) + "_" + str(sp)
            ms_file_name = g.default_repo_dir + "rlt/" + getMSName()

            if os.path.exists(ms_file_name):
                print(ms_file_name + " already generated. Continue.")
                return
            X = getDat(t, t.dataList, g.default_repo_dir +
                       "samples/Mix/M1" + hostname, t.featureType)
            dataList = [Ent(1, t.data, 'true', None)]
            Y = getDat(t, dataList, g.default_repo_dir +
                       "samples/Mix/M2" + hostname, t.featureType)

            modeScore = mode_score(X, Y)
            f = open(ms_file_name, "wb")
            pickle.dump(modeScore, f)
            f.close()

    elif t.mode == "FID":
        mkdir(g.default_repo_dir + "samples/Mix")
        mkdir(g.default_repo_dir + "rlt")
        for sp in range(0, 5):
            def getFidName():
                ans = 'fid_' + t.data + '_' + t.featureType + '_'
                for entry in t.dataList:
                    ans = ans + entry.data + '_' + entry.folder + '_' + \
                        ('%.2f' % entry.fraction) + '_' + \
                        str(entry.dup) + '_' + str(entry.imageMode) + '_'
                return ans + str(t.mixSize) + "_" + str(sp)
            fid_file_name = g.default_repo_dir + "rlt/" + getFidName()

            if os.path.exists(fid_file_name):
                print(fid_file_name + " already generated. Continue.")
                return
            X = getDat(t, t.dataList, g.default_repo_dir +
                       "samples/Mix/M1" + hostname, t.featureType)
            dataList = [Ent(1, t.data, 'true', None)]
            Y = getDat(t, dataList, g.default_repo_dir +
                       "samples/Mix/M2" + hostname, t.featureType)

            FID = fid(X, Y)
            f = open(fid_file_name, "wb")
            pickle.dump(FID, f)
            f.close()

    elif t.mode == 'collapse':
        run_collapse_exp(t.data)

    elif t.mode == 'drop':
        run_drop_exp(t.data)

    elif t.mode == 'overfit':
        run_overfit_exp(t.data)

    elif t.mode == 'Gen':  # generate necessary data..
        if t.data == 'noise':
            print("\nSampling noise...")
            noise_sampler(t)
        elif t.data == 'mnistsub':  # subclass
            print("\nSampling subclass...")
            t.data = 'mnist'
            subclass_sampler(t)
        else:
            t.folderName = t.model + \
                (str(t.epoch) if not t.model.startswith('true') else "")
            if t.model.startswith('true'):
                print("\nSampling " + t.folderName + " ...")
                folder_sampler(t)
            else:
                print("\nSampling " + t.folderName + " ...")
                generator_sampler(t)

    elif t.mode == "Gan":  # generate some fake images..
        if t.model == "DCGAN":
            DCGAN_main(t)
        elif t.model == "WGAN":
            WGAN_main(t)
        elif t.model == "NNGAN":
            NNGAN_main(t)
        elif t.model == "MGGAN":
            MGGAN_main(t)
        elif t.model == "DCGAN_cluster":
            DCGAN_cluster_main(t)
        else:
            print("error.. Unknown model.")

    else:
        print("not designed yet.")


ordering = np.random.permutation(len(tasks))
if opt.serial:
    ordering = range(0, len(tasks))

for i in ordering:
    # if not finished yet. otherwise just skip this task.
    if not os.path.isfile(labels + str(i) + "_Finished"):
        found = -1
        ts = tasks[i]
        for m in range(opt.A, opt.B + 1):  # all possible machines
            # this file was operated before
            if os.path.isfile(labels + str(i) + "M" + str(m)):
                found = m
                break

        if "M" + str(found) == hostname:  # was my task..
            print("found my unfinished task " + labels + str(i) + hostname)
            print("Will redo.")
            found = -1

        if found == -1:  # now let's do it!
            print("\nRunning at task " + str(i))
            with open(labels + str(i) + hostname, "w") as f:
                f.write("")
            run_combine(ts)
            with open(labels + str(i) + "_Finished", "w") as f:
                f.write("")
    else:
        print("task " + str(i) + " already finished.")
