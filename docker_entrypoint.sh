#!/bin/bash
set -e
figlet -c ANN 
python3.7 ~/breast_cancer_detection/ANN.py 2>/dev/null
figlet -c KNN
python3.7 ~/breast_cancer_detection/KNN.py 2>/dev/null
figlet -c SVM
python3.7 ~/breast_cancer_detection/SVM.py 2>/dev/null
