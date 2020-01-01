#!/bin/bash
set -e
exec figlet -c ANN  KNN  SVM & 
exec python3.7 ~/breast_cancer_detection/ANN.py &
exec python3.7 ~/breast_cancer_detection/KNN.py &
exec python3.7 ~/breast_cancer_detection/SVM.py
