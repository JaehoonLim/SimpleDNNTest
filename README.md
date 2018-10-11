# SimpleDNNTest
Simple package for DNN test with Keras

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
python convertROOTtoNumpy.py samples/Signal.root -t TEST_tree -b TEST_val1 TEST_val2 TEST_val3  
```

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
