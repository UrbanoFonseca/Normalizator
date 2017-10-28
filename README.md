# Normalizator
Python library for normalization of continuous variables.


### Installing
Clone the repository with git
```
git clone https://github.com/UrbanoFonseca/Normalizator
```

then install with the pip command on the destination folder
```
pip install .
```
To use in the script simply type
```
from normalizator import normalizator
```

### Normalizators
The package includes the following scalers:
* Standard (Z-Score)
* Min Max
* Decimal
* Median [1]
* Median and Median Absolute Deviation
* Max [2]
* Modified Tanh [2]

[1] Jayalakshmi, T. and Santhakumaran, A. (2011). Statistical Normalization and Back Propagation for Classification. International Journal of Computer Theory and Engineering, pp.89-93.

[2] S.Thangasamy and L.Latha (2011). Efficient approach to Normalization of Multimodal Biometric Scores. International Journal of Computer Applications 32(10), pp.57-64.
