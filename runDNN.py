"""
Run DNN - Using Keras API
Original Author: JaeHoon Lim (Department of Physics, Korea University jae.hoon.lim@cern.ch)
Created: 2018-Jun-20
"""

import sys
import argparse
import convertROOTtoNumpy
import Train
import Apply

def TestDNN():

    #signame = "samples/Signal.root"
    #bkgname = "samples/Background.root"
    #bkgflag = None
    # or
    signame = "samples/Allsample.root"
    bkgname = None
    bkgflag = "isSignal"

    treename = None # None : auto

    outputpath = "TEST_output/"
    weightpath = "TEST_weight/"

    #varlist = None # None : all variables
    varlist = ['TEST_val1','TEST_val2','TEST_val3']

    NNarch = [20, 10]
    validsample = 40.0 # persent (%)
    epoch = 100
    rdnseed = 22222222

    TestDNNwithArgs(signame, bkgname, bkgflag, treename, varlist, outputpath, weightpath, NNarch, validsample, epoch, rdnseed) 

def TestDNNwithArgs(signame, bkgname, bkgflag, treename, varlist, outputpath, weightpath, NNarch, validsample, epoch, rdnseed):

    if bkgflag is not None:
        allvarlist = varlist[:]    
        allvarlist.append(bkgflag)    
        convertROOTtoNumpy.processROOTfiles(signame,treename,allvarlist)
        Train.train_and_validate(signame, bkgname, bkgflag, varlist, weightpath, validsample, NNarch, rdnseed, epoch)
        Apply.processDNNfilelist(signame, outputpath, weightpath, treename)

    else :
        convertROOTtoNumpy.processROOTfiles(signame,treename,varlist)
        convertROOTtoNumpy.processROOTfiles(bkgname,treename,varlist)
        Train.train_and_validate(signame, bkgname, bkgflag, varlist, weightpath, validsample, NNarch, rdnseed, epoch)
        Apply.processDNNfilelist(signame, outputpath, weightpath, treename)
        Apply.processDNNfilelist(bkgname, outputpath, weightpath, treename)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Run without \'input root file (-i option)\' will read arguments in script')
    parser.add_argument('-i', type=str,help='\'I\'nput (signal) root file path')
    parser.add_argument('-b', type=str,help='\'B\'ackground root file path')
    parser.add_argument('-f', type=str,help='boolean \'F\'lag variable for signal & background (signal=true)')
    parser.add_argument('-t', type=str,help='\'T\'ree name')
    parser.add_argument('-v', type=str,nargs='+',help='\'V\'ariable list (default=all variables)')
    parser.add_argument('-o', type=str,default='./Output/',help='\'O\'uput file path (default=\'./Output/\')')
    parser.add_argument('-w', type=str,default='./TrainResult/',help='\'W\'eight file path (default=\'./TrainResult/\')')
    parser.add_argument('-a', type=int,default=[50,10],nargs='+',help='NN \'A\'rchitecture (default=[50,10])')
    parser.add_argument('-p', type=float,default=25.0,help='\'P\'ersent of validation sample (default=25.0)')
    parser.add_argument('-e', type=int,default=100,help='\'E\'poch (default=100)')
    parser.add_argument('-r', type=int,default=11111111,help='\'R\'andom seed number (default=11111111)')
    args = parser.parse_args()

    if args.i is not None:
        if args.b is None and args.f is None:
            print('ERROR : Need background sample : (file with -b option or flag with -f option)')
            sys.exit()
        elif args.b is not None and args.f is not None:
            print('ERROR : Use 1 background option : (file with -b option or flag with -f option)')
            sys.exit()
        TestDNNwithArgs(args.i,args.b,args.f,args.t,args.v,args.o,args.w,args.a,args.p,args.e,args.r)
    else:
        TestDNN()
