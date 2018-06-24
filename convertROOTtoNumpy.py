"""
Convert ROOT files into python numpy readable files
Original Author: Suyong Choi (Department of Physics, Korea University suyong@korea.ac.kr)
Created: 2018-Feb-12

Modified: JaeHoon Lim (jae.hoon.lim@cern.ch) / 2018-Jun-20
"""

import sys
import os
import re
import pickle
import argparse
from array import array
import numpy as np

def convertROOTtoNumpy(rootfile, treename=None, branchnamelist=None):

    from ROOT import TFile, TKey, TChain, TBranch #PyROOT hijacks command line arguments 

    # set output files
    numpyfile = rootfile.replace('.root', '.npy')
    branchnamelistfile = rootfile.replace('.root', '.pkl')

    # check tree name
    tf = TFile(rootfile)
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

    # open root file and get list of branches
    tc = TChain(treename)
    tc.Add(rootfile)

    tc.LoadTree(0) # you need to do this to load the file
    nentries = tc.GetEntries()
    
    allbranchlist = tc.GetListOfBranches()
    allbranchnamelist = map(TBranch.GetName, allbranchlist)
    branchindices = []
    branchlist = []       
    tmpvlist=[]
    # if no branchnamelist is given then process all branches
    # could run into problem if a branch is not a number type
    if branchnamelist is None:
        getall = 1
        ndim = len(allbranchnamelist)
        branchnamelist = allbranchnamelist
        branchindices = range(ndim)
    else:
        getall = 0        
        for abranch in branchnamelist:
            if abranch in allbranchnamelist: 
                index = allbranchnamelist.index(abranch)
                branchindices.append(index)
            else:
                print('ERROR : Check branch name : %s'%(abranch))
                sys.exit()
 
    # create raw list of branches to be used
    branchlist = map(allbranchlist.At, branchindices)
    valid_branchnamelist=[]
    # create variables that will store 
    for abranch in branchlist:
        leaves = abranch.GetListOfLeaves()
        branchname = abranch.GetName()
        for leaf in leaves:
            leaftype = leaf.GetTypeName()
            print(branchname + ' : ' + leaftype)
            validleaf = True
            if leaftype == 'Double_t':
                vtype = 'd'
            elif leaftype == 'Int_t':
                vtype = 'i'
            elif leaftype == 'Float_t':
                vtype = 'f'
            elif leaftype == 'Char_t':
                vtype = 'b'
            else:
                print(abranch.GetName()+' is not a number but of type %s'%leaftype)
                print('This branch will be skipped')
                validleaf = False
            
            if validleaf:
                # must declare array type to store the root leaf value 
                tmpvar = array(vtype, [0])
                
                valid_branchnamelist.append(branchname)
                tc.SetBranchStatus(branchname, 1)
                tc.SetBranchAddress(branchname, tmpvar)
                # this list contains the variables that will get updated when GetEntry is called
                tmpvlist.append(tmpvar)
                
    # number of leaves that are really numbers
    ndim = len(tmpvlist)
    
    try:
        assert(ndim == len(tmpvlist))
    except:
        print('Total branches %d'%ndim)
        print('Allocated variables %d'%(len(tmpvlist)))
        raise RuntimeError('variable list does not match the number of branches')

    # create numpy
    convertednumpy = np.zeros(shape=(nentries, ndim), dtype=np.float32)
    
    for iev in range(nentries):
        tc.GetEntry(iev, getall)
        for ivar in range(ndim):
            convertednumpy[iev, ivar] = tmpvlist[ivar][0]*1.0
        if iev%1000==0 and iev > 0:
            print('Read entry %d'%(iev))
    np.save(numpyfile, convertednumpy)
    pickle.dump(valid_branchnamelist, open(branchnamelistfile, 'wt'))
    #print('Wrote %d entries from %s into %s file'%(nentries, rootfile, numpyfile))
    if nentries>0:
        print('Wrote %d entries from %s'%(nentries, rootfile))
    else :
        print('WARNING : ZERO entries files will return ERROR : %s'%(rootfile))

def processROOTfiles(inputfiles, treename, branchnamelist):
    if os.path.isdir(inputfiles):
        flist = os.listdir(inputfiles)
        for fname in flist:
            fullname = os.path.join(inputfiles, fname)
            if os.path.isfile(fullname) and re.match('.*\.root', fname):
                convertROOTtoNumpy(fullname, treename, branchnamelist)
            elif os.path.isdir(fname):
                processROOTfiles(fullname, treename, branchnamelist)
    elif os.path.isfile(inputfiles):
        if re.match('.*\.root', inputfiles):
            convertROOTtoNumpy(inputfiles, treename, branchnamelist)
        else :
            print('ERROR : Check input root file name : %s'%(inputfiles))
            sys.exit()
    else :
        print('ERROR : Check input path : %s'%(inputfiles))
        sys.exit()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('I', type=str,help='\'I\'nput root file or path')
    parser.add_argument('-t', type=str,help='\'T\'ree name')
    parser.add_argument('-b', type=str,nargs='+',help='\'B\'ranch name')
    args = parser.parse_args()

    processROOTfiles(args.I, args.t, args.b) 
