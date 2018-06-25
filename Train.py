"""
Training - Using Keras API
Original Author:  Suyong Choi (Department of Physics, Korea University suyong@korea.ac.kr)
Created: 2018-Feb-22

Modified: JaeHoon Lim (jae.hoon.lim@cern.ch) / 2018-Jun-20
"""

import sys
import os
import re
import math
import pickle
import argparse
from array import array
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

def findcolumns(varlist, columnname):
    """
    [summary]
    Arguments:
        varlist {string list} -- Variable list of interest
        columnname {string list} -- All variables. column names of numpy data read in
    
    Returns:
        list of indices -- column positions corresponding to the variable list
    """
    indices = []
    for v in varlist:
        if v in columnname:
            indices.append(columnname.index(v))
        else:
            print('ERROR : Check variable name : %s'%(v))
            sys.exit()
    return indices

# call back function to store history of losses
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def train_and_validate(sigfile, bkgfile=None, bkgflag=None, varlist=None, trainoutputdir="./TrainResult/", valid_persent=25, netarch=[50,10], rndseed=11111111, nruns=100, minibatchsize=128, retrain=False, DrawPlot=True, PrintLog=2):

    from ROOT import gStyle, TCanvas, TH1D, TGraph, TLine, TLegend, TText #PyROOT hijacks command line arguments 

    # check signal file
    if not re.match('.*\.npy', sigfile):
        if re.match('.*\.root', sigfile):
            sigfile = sigfile.replace('.root','.npy')
            print('Signal numpy file : load npy instead of root')
        elif re.match('.*\.pkl', sigfile):
            sigfile = sigfile.replace('.pkl','.npy')
            print('Signal numpy file : load npy instead of pkl')
    if not os.path.isfile(sigfile):
        print('ERROR : Check signal file : %s'%(sigfile))
        sys.exit()

    # check background file
    if bkgflag is not None:
        bkgfile = sigfile
    if not re.match('.*\.npy', bkgfile):
        if re.match('.*\.root', bkgfile):
            bkgfile = bkgfile.replace('.root','.npy')
            print('Background numpy file : load npy instead of root')
        elif re.match('.*\.pkl', bkgfile):
            bkgfile = bkgfile.replace('.pkl','.npy')
            print('Background numpy file : load npy instead of npy')
    if not os.path.isfile(bkgfile):
        print('ERROR : Check background file : %s'%(bkgfile))
        sys.exit()

    # read signal numpy arrays
    sig = np.load(sigfile)
    # read corresponding map of variable name and column indices
    sigfile = sigfile.replace('.npy','.pkl')
    if not os.path.isfile(sigfile):
        print('ERROR : Check signal file : %s'%(sigfile))
        sys.exit()
    sigcolumns = pickle.load(open(sigfile, 'rb'))
    
    # read background numpy arrays
    bkg = np.load(bkgfile)
    bkgfile = bkgfile.replace('.npy','.pkl')
    if not os.path.isfile(bkgfile):
        print('ERROR : Check background file : %s'%(bkgfile))
        sys.exit()
    bkgcolumns = pickle.load(open(bkgfile, 'rb'))

    # variable list
    if varlist is None:
        varlist = sigcolumns

    # weight file
    save_dir = trainoutputdir 
    if os.path.exists('{}/weights.hdf5'.format(save_dir)):
        print('WARNING : Old weight file has removed')
        os.remove('{}/weights.hdf5'.format(save_dir))

    # boolean flag for signal & background
    if bkgflag is not None:
        sig_check = sigcolumns.index(bkgflag)
        bkg_check = bkgcolumns.index(bkgflag)

        is_sig = sig[0::, sig_check]!=0
        is_bkg = bkg[0::, bkg_check]==0

        sig = sig[is_sig]
        bkg = bkg[is_bkg]

    # feature inputs for neural network
    sig_selectedcolumns = findcolumns(varlist, sigcolumns)
    bkg_selectedcolumns = findcolumns(varlist, bkgcolumns)

    sig_inputs = np.take(sig, sig_selectedcolumns, axis=1)
    bkg_inputs = np.take(bkg, bkg_selectedcolumns, axis=1)

    nsignal = sig_inputs.shape[0]
    nbackground = bkg_inputs.shape[0]
    ntotal = nsignal + nbackground

    inputs_unshuffled = np.vstack((sig_inputs, bkg_inputs))
    outputs_unshuffled = np.reshape(np.array([1,0]*nsignal + [0,1]*nbackground), (nsignal+nbackground, 2))
   
    # normalize each data column to have 0 mean and 1 stddev
    # Otherwise, the NN won't train properly
    # calculate mean and stddev along each column
    inputmeans = np.mean(inputs_unshuffled, axis=0)
    inputsigma = np.std(inputs_unshuffled, axis=0)

    # normalized inputs
    normedinputs_unshuffled = (inputs_unshuffled - inputmeans)/inputsigma
    normed_signal = normedinputs_unshuffled[0:nsignal, 0::]
    normed_background = normedinputs_unshuffled[nsignal:nsignal+nbackground, 0::]

    # shuffle
    np.random.seed(rndseed)
    randorder = np.random.permutation(ntotal)
    normedinputs = normedinputs_unshuffled[randorder, 0::]
    twoclassoutputs = outputs_unshuffled[randorder]

    # divide sample
    train_normedinputs = normedinputs[int(ntotal*valid_persent/100)+1::, 0::]
    train_twoclassoutputs = twoclassoutputs[int(ntotal*valid_persent/100)+1::]
    valid_normedinputs = normedinputs[0:int(ntotal*valid_persent/100):, 0::]
    valid_twoclassoutputs = twoclassoutputs[0:int(ntotal*valid_persent/100):]

    is_train_bkg = train_twoclassoutputs[0::,1]==1
    is_train_sig = train_twoclassoutputs[0::,0]==1
    is_valid_bkg = valid_twoclassoutputs[0::,1]==1
    is_valid_sig = valid_twoclassoutputs[0::,0]==1

    print('\nValidation Sample : %d %%'%(valid_persent))
    print('Total Signal Sample : %d\t\tTotal Background Sample : %d'%(nsignal, nbackground))
    print('Train Signal Sample : %d\t\tTrain Background Sample : %d\t\tTotal Train Sample : %d'%(train_twoclassoutputs[is_train_sig].shape[0], train_twoclassoutputs[is_train_bkg].shape[0],train_twoclassoutputs[is_train_sig].shape[0]+train_twoclassoutputs[is_train_bkg].shape[0]))
    print('Valid Signal Sample : %d\t\tValid Background Sample : %d\t\tTotal Valid Sample : %d\n'%(valid_twoclassoutputs[is_valid_sig].shape[0], valid_twoclassoutputs[is_valid_bkg].shape[0],valid_twoclassoutputs[is_valid_sig].shape[0]+valid_twoclassoutputs[is_valid_bkg].shape[0]))

    # number of feature inputs
    ndim = len(varlist)

    # define classifier using Keras
    classifier = Sequential()

    # define NN using KERAS
    alpha=1e-5
    for i in range(len(netarch)):
        if i==0:
            inputdim = ndim
        else:
            inputdim = netarch[i-1]
        classifier.add(Dense(netarch[i], input_dim=inputdim
            , activation='relu'
            , kernel_regularizer=regularizers.l2(alpha)
            , activity_regularizer=regularizers.l2(alpha)
            , kernel_initializer='glorot_uniform'))
        classifier.add(BatchNormalization())
    classifier.add(Dense(2, activation='softmax'
        , kernel_regularizer=regularizers.l2(alpha)
        , kernel_initializer='glorot_uniform'))

    optim_adagrad = keras.optimizers.adagrad(lr=0.01)
    classifier.compile(optimizer=optim_adagrad, loss='categorical_crossentropy', metrics=['accuracy'])

    # save weights
    modelfname = os.path.join(save_dir, 'weights.hdf5')
    if os.path.exists(save_dir):
        if os.path.exists(modelfname) and not retrain:
            keras.models.load_model(modelfname)
    else:
        os.makedirs(save_dir)

    checkpointer = ModelCheckpoint(filepath=modelfname, verbose=1, save_best_only=False)

    # save input variables
    fhandle = open(os.path.join(save_dir, 'inputvars.pkl'), mode='wb')
    pickle.dump(varlist,  fhandle)
    # save mean and rms - needed to normalize data when applying
    pickle.dump((inputmeans, inputsigma),  open(os.path.join(save_dir, 'inputmeanrms.pkl'), mode='wb'))

    # train
    early_stopping = EarlyStopping(patience = 10)
    test = classifier.fit(train_normedinputs, train_twoclassoutputs, epochs=nruns, batch_size=128, verbose=PrintLog, validation_data=(valid_normedinputs,valid_twoclassoutputs), callbacks=[early_stopping,checkpointer])

    # plot results
    if DrawPlot:

        import matplotlib.pyplot as plt
        gStyle.SetOptStat(0) 

        # loss plot
        n_test = len(test.history['loss'])
        c_loss = TCanvas('c_loss','Loss',800,600)
        h_loss = TH1D('h_loss','Model Loss;Epoch;Loss',n_test+1,0,n_test+1)
        t_loss = TGraph(n_test, array('d',np.add(range(n_test),1.0)), array('d',test.history['loss']))
        t_val_loss = TGraph(n_test, array('d',np.add(range(n_test),1.0)), array('d',test.history['val_loss']))

        l_loss = TLegend(0.75,0.8,0.9,0.9)
        l_loss.AddEntry(t_loss,"   Train","l")
        l_loss.AddEntry(t_val_loss,"   Valid","l")

        max_loss = max(max(test.history['loss']), max(test.history['val_loss']))
        if max_loss > 2:
            max_loss = 1.5*round(max_loss/(10**(int(math.log10(max_loss)))))*(10**(int(math.log10(max_loss))))
        else:
            max_loss = 1.5*round(max_loss/(10**(int(math.log10(max_loss))-1)))*(10**(int(math.log10(max_loss))-1))
        h_loss.SetMaximum(max_loss)

        t_loss.SetLineColor(8)
        t_loss.SetLineWidth(2)
        t_val_loss.SetLineColor(95)
        t_val_loss.SetLineWidth(2)
        t_val_loss.SetLineStyle(2)

        h_loss.DrawCopy()
        t_loss.Draw('same')
        t_val_loss.Draw('same')
        l_loss.Draw('same')

        c_loss_out = os.path.join(save_dir, 'Loss.png')
        c_loss.SaveAs(c_loss_out)

        # over-train plot
        overtrain_train_sig = classifier.predict(train_normedinputs[is_train_sig], batch_size=len(train_normedinputs[is_train_sig]))
        overtrain_train_bkg = classifier.predict(train_normedinputs[is_train_bkg], batch_size=len(train_normedinputs[is_train_bkg]))
        overtrain_valid_sig = classifier.predict(valid_normedinputs[is_valid_sig], batch_size=len(valid_normedinputs[is_valid_sig]))
        overtrain_valid_bkg = classifier.predict(valid_normedinputs[is_valid_bkg], batch_size=len(valid_normedinputs[is_valid_bkg]))

        c_overtrain = TCanvas('c_overtrain','Over-Train Check',800,600)
        h_train_sig = TH1D('h_train_sig','Over-Train Check;DNN;Normalized to Unity',44,-0.05,1.05)
        h_train_sig.Sumw2() # for ROOT5
        for t_sig in (overtrain_train_sig[0::,0].reshape(overtrain_train_sig.shape[0])):
            h_train_sig.Fill(t_sig)
        h_train_bkg = TH1D('h_train_bkg','Over-Train Check;DNN;Normalized to Unity',44,-0.05,1.05)
        h_train_bkg.Sumw2() # for ROOT5
        for t_bkg in overtrain_train_bkg[0::,0].reshape(overtrain_train_bkg.shape[0]):
            h_train_bkg.Fill(t_bkg)
        h_valid_sig = TH1D('h_valid_sig','Over-Train Check;DNN;Normalized to Unity',44,-0.05,1.05)
        h_valid_sig.Sumw2() # for ROOT5
        for v_sig in overtrain_valid_sig[0::,0].reshape(overtrain_valid_sig.shape[0]):
            h_valid_sig.Fill(v_sig)
        h_valid_bkg = TH1D('h_valid_bkg','Over-Train Check;DNN;Normalized to Unity',44,-0.05,1.05)
        h_valid_bkg.Sumw2() # for ROOT5
        for t_bkg in overtrain_valid_bkg[0::,0].reshape(overtrain_valid_bkg.shape[0]):
            h_valid_bkg.Fill(t_bkg)

        h_valid_sig.Scale(1./h_valid_sig.Integral())
        h_valid_sig.SetFillColor(9)
        h_valid_sig.SetLineColor(9)
        h_valid_sig.SetLineWidth(2)

        h_valid_bkg.Scale(1./h_valid_bkg.Integral())
        h_valid_bkg.SetFillColor(2)
        h_valid_bkg.SetLineColor(2)
        h_valid_bkg.SetLineWidth(2)
        h_valid_bkg.SetFillStyle(3004)

        h_train_sig.Scale(1./h_train_sig.Integral())
        h_train_sig.SetFillColor(4)
        h_train_sig.SetLineColor(4)
        h_train_sig.SetLineWidth(2)
        h_train_sig.SetMarkerStyle(21)
        h_train_sig.SetMarkerSize(0.7)
        h_train_sig.SetMarkerColor(4)

        h_train_bkg.Scale(1./h_train_bkg.Integral())
        h_train_bkg.SetFillColor(2)
        h_train_bkg.SetLineColor(2)
        h_train_bkg.SetLineWidth(2)
        h_train_bkg.SetMarkerStyle(21)
        h_train_bkg.SetMarkerSize(0.7)
        h_train_bkg.SetMarkerColor(2)

        hist_max = max([h_train_sig.GetMaximum(), h_train_bkg.GetMaximum(), h_valid_sig.GetMaximum(), h_valid_bkg.GetMaximum()])
        h_valid_sig.SetMaximum(1.4*hist_max)

        l_plot = TLegend(0.6,0.7,0.9,0.9)
        l_plot.AddEntry(h_train_sig,"Train Sample - Signal : {:.0f}".format(h_train_sig.GetEntries()),"lpe")
        l_plot.AddEntry(h_train_bkg,"Train Sample - Background : {:.0f}".format(h_train_bkg.GetEntries()),"lpe")
        l_plot.AddEntry(h_valid_sig,"Valid Sample - Signal : {:.0f}".format(h_valid_sig.GetEntries()),"f")
        l_plot.AddEntry(h_valid_bkg,"Valid Sample - Background : {:.0f}".format(h_valid_bkg.GetEntries()),"f")

        h_valid_sig.DrawCopy('hist')
        h_valid_bkg.DrawCopy('histsame')
        h_train_sig.DrawCopy('pe1same')
        h_train_bkg.DrawCopy('pe1same')
        l_plot.Draw('same')

        c_overtrain_out = os.path.join(save_dir, 'OverTrain.png')
        c_overtrain.SaveAs(c_overtrain_out)

        # roc plot
        c_roc = TCanvas('c_roc','ROC Curve',800,600)
        h_roc_sig = TH1D('h_roc_sig','ROC Curve;DNN;Normalized to Unity',1000,0.0,1.0)
        h_roc_bkg = TH1D('h_roc_bkg','ROC Curve;DNN;Normalized to Unity',1000,0.0,1.0)
        h_roc = TH1D('h_roc','ROC Curve;Signal efficiency;Background rejection rate',22,-0.05,1.05)
        for dummy in range(h_roc.GetNbinsX()+2):
            h_roc.SetBinContent(dummy,-1)
        h_roc.SetMaximum(1.05)
        h_roc.SetMinimum(-0.05)
        for t_sig in (overtrain_train_sig[0::,0].reshape(overtrain_train_sig.shape[0])):
            h_roc_sig.Fill(t_sig)
        for v_sig in overtrain_valid_sig[0::,0].reshape(overtrain_valid_sig.shape[0]):
            h_roc_sig.Fill(v_sig)
        for t_bkg in overtrain_train_bkg[0::,0].reshape(overtrain_train_bkg.shape[0]):
            h_roc_bkg.Fill(t_bkg)
        for t_bkg in overtrain_valid_bkg[0::,0].reshape(overtrain_valid_bkg.shape[0]):
            h_roc_bkg.Fill(t_bkg)
        h_roc_sig.Scale(1./h_roc_sig.Integral())
        h_roc_bkg.Scale(1./h_roc_bkg.Integral())
        sig_eff = []
        bkg_rej = []
        sig_90 = 0
        sig_95 = 0
        bkg_90 = 0
        bkg_95 = 0
        for i_roc in range(1002):
            sig_eff.append(1.0-h_roc_sig.Integral(0,i_roc))
            bkg_rej.append(h_roc_bkg.Integral(0,i_roc))
            if (1.0-h_roc_sig.Integral(0,i_roc)) > 0.90:
                sig_90 = h_roc_bkg.Integral(0,i_roc)
            if (1.0-h_roc_sig.Integral(0,i_roc)) > 0.95:
                sig_95 = h_roc_bkg.Integral(0,i_roc)
            if (h_roc_bkg.Integral(0,i_roc)) < 0.90:
                bkg_90 = 1.0-h_roc_sig.Integral(0,i_roc)
            if (h_roc_bkg.Integral(0,i_roc)) < 0.95:
                bkg_95 = 1.0-h_roc_sig.Integral(0,i_roc)

        t_roc = TGraph(1002, array('d',sig_eff), array('d',bkg_rej))

        h_roc.DrawCopy()
        t_roc.SetLineWidth(2)
        t_roc.Draw('same')

        l_roc = TLine()
        l_roc.SetLineWidth(2)
        l_roc.SetLineStyle(2)
        l_roc.DrawLine(0.90,0.05,0.90,sig_90)
        l_roc.DrawLine(-0.05,sig_90,0.90,sig_90)
        l_roc.DrawLine(0.95,0.05,0.95,sig_95)
        l_roc.DrawLine(-0.05,sig_95,0.95,sig_95)
        l_roc.DrawLine(0.05,0.90,bkg_90,0.90)
        l_roc.DrawLine(bkg_90,-0.05,bkg_90,0.90)
        l_roc.DrawLine(0.05,0.95,bkg_95,0.95)
        l_roc.DrawLine(bkg_95,-0.05,bkg_95,0.95)

        x_roc = TText()
        x_roc.SetTextAlign(11)
        x_roc.SetTextFont(43)
        x_roc.SetTextSize(15)
        x_roc.DrawText(-0.02,sig_90+0.01,"%.3f"%(sig_90))
        x_roc.DrawText(0.06,sig_95+0.01,"%.3f"%(sig_95))
        x_roc.DrawText(bkg_90+0.01,0.06,"%.3f"%(bkg_90))
        x_roc.DrawText(bkg_95+0.01,0.00,"%.3f"%(bkg_95))

        c_roc_out = os.path.join(save_dir, 'ROC_curve.png')
        c_roc.SaveAs(c_roc_out)

        plt.show()
        raw_input("\nPress Enter to exit...\n")  

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('I', type=str,help='\'I\'nput (signal) numpy file path')
    parser.add_argument('-b', type=str,help='\'B\'ackground numpy file path')
    parser.add_argument('-f', type=str,help='boolean \'F\'lag variable for signal & background (signal=true)')
    parser.add_argument('-a', type=int,default=[50,10],nargs='+',help='NN \'A\'rchitecture (default=[50,10])')
    parser.add_argument('-p', type=float,default=25.0,help='\'P\'ersent of validation sample (default=25.0)')
    parser.add_argument('-w', type=str,default='./TrainResult/',help='\'W\'eight file path (default=\'./TrainResult/\')')
    parser.add_argument('-v', type=str,nargs='+',help='\'V\'ariable list (default=all variables)')
    parser.add_argument('-e', type=int,default=100,help='\'E\'poch (default=100)')
    parser.add_argument('-r', type=int,default=11111111,help='\'R\'andom seed number (default=11111111)')
    args = parser.parse_args()

    if args.b is None and args.f is None:
        print('ERROR : Need background sample : (file with -b option or flag with -f option)')
        sys.exit()
    elif args.b is not None and args.f is not None:
        print('ERROR : Use 1 background option : (file with -b option or flag with -f option)')
        sys.exit()

    train_and_validate(sigfile=args.I, bkgfile=args.b, bkgflag=args.f, trainoutputdir=args.w, valid_persent=args.p, varlist=args.v, netarch=args.a, rndseed=args.r, nruns=args.e) 
