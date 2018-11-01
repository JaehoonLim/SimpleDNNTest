# SimpleDNNTest
This package will help whom using __ROOT__ Ntuple to run DNN test with __Keras__.  
  
- __Keras__ : https://keras.io  
- __ROOT__ : https://root.cern.ch  

# Acknowledge
This work was suppored by Global Science experimental Data hub Center (GSDC) in Korea Institute of Science and Technology Information (KISTI).

# 0. Enviroment

- __ROOT__ >= 5.34 (include __ROOT 6__)  
- __Python__ >= 2.7 (include __Python 3__)  
- __TensorFlow__ >= 1.4.0 rc0  

And __h5py__, __matplotlib__  
  
# 1. convertROOTtoNumpy.py  
```
usage: convertROOTtoNumpy.py [-h] [-t T] [-b B [B ...]] I  
  
positional arguments:  
   I           'I'nput root file or path  
  
optional arguments:  
  -h, --help    show this help message and exit  
  -t           'T'ree name  
  -b           'B'ranch name  
```    
__ex)__
```
python mkSampleRootFile.py  
python convertROOTtoNumpy.py samples/Signal.root -t TEST_tree -b TEST_val1 TEST_val2 TEST_val3  
```

For __TensorFlow__, we will convert __ROOT__ Ntuple file to __NumPy__ file.  
  
__convertROOTtoNumpy.py__ needs __ROOT__ Ntuple file as input file. When __convertROOTtoNumpy.py__ run finished, it will give you 2 kinds of output files.  
First one, __NumPy__ array file(__.npy__), contains varialbes you selected form __ROOT__ Ntuple for __TensorFlow__. Second one, __Python pickle__ file(__.pkl__), contains the name of varialbes with column index. You can check the variable name of __NumPy__ array by using this pickle file.  
  
without '-t' option, __convertROOTtoNumpy.py__ will automatically find a tree in __ROOT__ Ntuple.  
without '-b' option, all branches in the tree will be converted.  
  
# 2. Train.py  
```
usage: Train.py [-h] [-b B] [-f F] [-a A [A ...]] [-p P] [-w W] [-v V [V ...]] [-e E] [-r R] I  
  
positional arguments:  
   I           'I'nput (signal) numpy file path  
  
optional arguments:  
  -h, --help   show this help message and exit  
  -b           'B'ackground numpy file path  
  -f           boolean 'F'lag variable for signal & background (signal=true)  
  -a           NN 'A'rchitecture (default=[50,10])  
  -p           'P'ersent of validation sample (default=25.0)  
  -w           'W'eight file path (default='./TrainResult/')  
  -v           'V'ariable list (default=all variables)  
  -e           'E'poch (default=100)  
  -r           'R'andom seed number (default=11111111)  
```

__ex)__    
```
python Train.py samples/Signal.npy -b samples/Background.npy -a 20 10 -p 40.0 -w TEST_weight -v TEST_val1 TEST_val2 TEST_val3  
```
or
```
python Train.py samples/Allsample.npy -f isSignal -a 20 10 -p 40.0 -w TEST_weight -v TEST_val1 TEST_val2 TEST_val3  
```  
  
__Train.py__ study characteristics of signal what we want and background what we don't want.  
  
As you can see on example, input can be 2 __NumPy__ files or 1 __NumPy__ file with boolean flag which signal is True.  
__Train.py__ is using the Multi-Layer Perceptron (MLP) model. You can set the MLP model, the number of layers and the number of nodes for each layer, by '-a' option. '-a 20 10' means 2 layers with 20 and 10 nodes for first and second layer, respectively.  
Before start training, __Train.py__ will divide samples to test sample and validation sample. Test sample will be used to training, and validation sample will be used to check over-training. Base on validation sample's accuracy, training will be stoped automatically for prevent over-training.  
After training, __Train.py__ will give you 3 plots and train weight file. With __Loss__ plot, you can check loss function results of test sample and validation sample. With __Over-Train Check__ plot, you can check DNN discriminator values of each samples. With __Receiver Operating Characteristic (ROC) Curve__ plot, you can check signal efficieny and background rejection rate.  
　　
# 3. Apply.py  
```
usage: Apply.py [-h] [-o O] [-w W] [-t T] I  
  
positional arguments:  
   I         'I'nput root file or path  
  
optional arguments:  
  -h, --help show this help message and exit  
  -o         'O'uput file path (default: ./Output/)  
  -w         'W'eight file path (default: ./TrainResult/)  
  -t         'T'ree name (default: None)  
```

__ex)__
```
python Apply.py samples/Signal.root -o TEST_output -w TEST_weight -t TEST_tree  
```
  
With train weight file form __Train.py__, __Apply.py__ will start last phase of DNN test.  
  
Based on training result, __Apply.py__ will give you a __ROOT__ Ntulple file same as you input with DNN test result. You can see the variable __'DNNValue'__ on output __ROOT__ file.  
  
# 4. runDNN.py  
```
usage: runDNN.py [-h] [-i I] [-b B] [-f F] [-t T] [-v V [V ...]] [-o O] [-w W] [-a A [A ...]] [-p P] [-e E] [-r R]  
  
Run without 'input root file (-i option)' will read arguments in script  
  
optional arguments:  
  -h, --help   show this help message and exit  
  -i           'I'nput (signal) root file path   
  -b           'B'ackground root file path  
  -f           boolean 'F'lag variable for signal & background (signal=true)  
  -t           'T'ree name  
  -v           'V'ariable list (default=all variables)  
  -o           'O'uput file path (default='./Output/')  
  -w           'W'eight file path (default='./TrainResult/')  
  -a           NN 'A'rchitecture (default=[50,10])  
  -p           'P'ersent of validation sample (default=25.0)  
  -e           'E'poch (default=100)  
  -r           'R'andom seed number (default=11111111)  
```

__ex) with ‘-i’ option__   
```
python runDNN.py -i samples/Signal.root -b samples/Background.root -a 20 10 -p 40.0 -o TEST_output -w TEST_weight -v TEST_val1 TEST_val2 TEST_val3  
```    
__without ‘-i’ option__
```
python runDNN.py  
```
  
__runDNN.py__ will help you to run all test step above at once.  
  
__runDNN.py__ 
