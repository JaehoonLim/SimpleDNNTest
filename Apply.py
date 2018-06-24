"""
Application - Using Keras API
Original Author: Suyong Choi (Department of Physics, Korea University suyong@korea.ac.kr)
Created: 2018-Mar-09

Modified: JaeHoon Lim (jae.hoon.lim@cern.ch) / 2018-Jun-20
"""

import os
import sys
import re
import pickle
import argparse
from array import array
import numpy as np
import keras
from Train import findcolumns

def apply(inputrootfile, outputdirectory="./Output/", trainednetdir="./TrainResult/", treename=None):
   
    from ROOT import TFile, TKey, TChain, TBranch #PyROOT hijacks command line arguments 

    # read signal numpy arrays
    inputnpyfile = inputrootfile.replace('.root', '.npy')
    if not os.path.isfile(inputnpyfile):
        print('ERROR : Check numpy file : %s'%(inputnpyfile))
        sys.exit()
    data = np.load(inputnpyfile)

    # read corresponding map of variable name and column indices
    columnnamef = inputrootfile.replace('.root', '.pkl')            
    if not os.path.isfile(columnnamef):
        print('ERROR : Check pickle file : %s'%(columnnamef))
        sys.exit()
    columns = pickle.load(open(columnnamef, 'rb'))
    
    # feature inputs for neural network
    inputvarfile = os.path.join(trainednetdir, 'inputvars.pkl')
    if not os.path.isfile(inputvarfile):
        print('ERROR : Check train result file : %s'%(inputvarfile))
        sys.exit()
    varlist = pickle.load(open(inputvarfile, mode='rb'))
    selectedcolumns = findcolumns(varlist, columns)
    inputs = np.take(data, selectedcolumns, axis=1)

    inputmeanfile = os.path.join(trainednetdir, 'inputmeanrms.pkl')
    if not os.path.isfile(inputmeanfile):
        print('ERROR : Check train result file : %s'%(inputmeanfile))
        sys.exit()
    (inputmeans, inputsigma) = pickle.load(open(inputmeanfile, mode='rb') )
    
    # normalize each data column to have 0 mean and 1 stddev
    # Otherwise, the NN won't train properly
    # calculate mean and stddev along each column

    # normalized inputs
    normedinputs = (inputs - inputmeans)/inputsigma

    modelfname = os.path.join(trainednetdir, 'weights.hdf5')
    if not os.path.isfile(modelfname):
        print('ERROR : Check train result file : %s'%(modelfname))
        sys.exit()
    classifier = keras.models.load_model(modelfname)

    yout = classifier.predict(normedinputs, batch_size=len(normedinputs))

    # check tree name
    tf = TFile(inputrootfile)
    tk = tf.GetListOfKeys()
    checkTree = False
    for key in tk:
        if TKey.GetClassName(key) == "TTree":
            if treename is None:
                treename = TKey.GetName(key)
                checkTree = True
            else:
                if treename == TKey.GetName(key):
                    checkTree = True
    if not checkTree:
        print('ERROR : Check tree name : %s'%(treename))
        sys.exit()

    tc = TChain(treename)
    tc.Add(inputrootfile)
    nentries = tc.GetEntries()

    # save outputs
    _, rootfilename = os.path.split(inputrootfile)
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    outputrootfile = os.path.join(outputdirectory, rootfilename)

    tfile = TFile(outputrootfile, 'RECREATE')
    tcout = tc.CloneTree();
    dnnout = array('f', [0.0])
    branch = tcout.Branch("DNNValue", dnnout, "DNNValue/F")
    for i in range(nentries):
        dnnout[0] = yout[i, 0]
        branch.Fill()
    tcout.Write()
    tfile.Close()
    print('DNNValue saved : %s'%(outputrootfile))

def processDNNfilelist(inputdirectory, outputdirectory, trainednetdir, treename):

    if os.path.isdir(inputdirectory):
        flist = os.listdir(inputdirectory)
        for fname in flist:
            fullname = os.path.join(inputdirectory, fname)
            if os.path.isfile(fullname) and re.match('.*\.root', fname):
                apply(fullname, outputdirectory, trainednetdir, treename)
            elif os.path.isdir(fullname):
                processDNNfilelist(fullname, outputdirectory, trainednetdir, treename)
    elif os.path.isfile(inputdirectory):
        if re.match('.*\.root', inputdirectory):
            apply(inputdirectory, outputdirectory, trainednetdir, treename)
        else :
            print('ERROR : Check input root file name : %s'%(inputdirectory))
            sys.exit()
    else :
        print('ERROR : Check input path : %s'%(inputdirectory))
        sys.exit()

if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('I', type=str,help='\'I\'nput root file or path')
    parser.add_argument('-o', type=str,default='./Output/',help='\'O\'uput file path')
    parser.add_argument('-w', type=str,default='./TrainResult/',help='\'W\'eight file path')
    parser.add_argument('-t', type=str,help='\'T\'ree name')
    args = parser.parse_args()

    processDNNfilelist(args.I, args.o, args.w, args.t)
