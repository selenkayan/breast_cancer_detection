#!/bin/bash
set -e
figlet -c ANN 
python3.7 ~/breast_cancer_detection/ANN.py
figlet -c KNN
python3.7 ~/breast_cancer_detection/KNN.py
figlet -c SVM
python3.7 ~/breast_cancer_detection/SVM.py
