"""
Make sample root file
Original Author: JaeHoon Lim (Department of Physics, Korea University jae.hoon.lim@cern.ch)
Created: 2018-Jun-20
"""

import os
from ROOT import TMath, TFile, TTree, TRandom, TF1, TBranch
from array import array

def PoissonReal(k, lambd):
    return TMath.Exp(k[0]*TMath.Log(lambd[0])-lambd[0]) / TMath.Gamma(k[0]+1.)

def mkSampleRootFile():

    if not os.path.exists('samples'):
        os.makedirs('samples')
    f_sig = TFile("samples/Signal.root","RECREATE")
    t_sig = TTree("TEST_tree","TEST_tree for SIG")
    f_bkg = TFile("samples/Background.root","RECREATE")
    t_bkg = TTree("TEST_tree","TEST_tree for BKG")
    f_all = TFile("samples/Allsample.root","RECREATE")
    t_all = TTree("TEST_tree","TEST_tree for ALL")

    rnd = TRandom()
    MyPoisson = TF1("MyPoisson", PoissonReal,0.,50.0,1)

    val1 = array('d', [0.0])
    val2 = array('d', [0.0])
    val3 = array('d', [0.0])
    isSignal = array('b', [True])

    t_sig.Branch("TEST_val1",val1,"TEST_val1/D")
    t_sig.Branch("TEST_val2",val2,"TEST_val2/D")
    t_sig.Branch("TEST_val3",val3,"TEST_val3/D")
    t_sig.Branch("isSignal",isSignal,"isSignal/B")

    t_bkg.Branch("TEST_val1",val1,"TEST_val1/D")
    t_bkg.Branch("TEST_val2",val2,"TEST_val2/D")
    t_bkg.Branch("TEST_val3",val3,"TEST_val3/D")
    t_bkg.Branch("isSignal",isSignal,"isSignal/B")

    t_all.Branch("TEST_val1",val1,"TEST_val1/D")
    t_all.Branch("TEST_val2",val2,"TEST_val2/D")
    t_all.Branch("TEST_val3",val3,"TEST_val3/D")
    t_all.Branch("isSignal",isSignal,"isSignal/B")

    MyPoisson.SetParameter(0,3.0)

    for i_sig in range(5000):
        val1[0] = rnd.Gaus(3.0,4.0)
        val2[0] = MyPoisson.GetRandom()
        val3[0] = rnd.Gaus(3.0,3.0)
        t_sig.Fill()
        t_all.Fill()


    MyPoisson.SetParameter(0,7.0)
    isSignal[0] = False

    for i_bkg in range(5000):
        val1[0] = rnd.Gaus(5.0,5.0)
        val2[0] = MyPoisson.GetRandom()
        val3[0] = rnd.Gaus(4.0,3.0)
        t_bkg.Fill()
        t_all.Fill()

    f_sig.Write();
    f_bkg.Write();
    f_all.Write();

if __name__=='__main__':
    mkSampleRootFile()

