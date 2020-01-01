#!/bin/bash
set -e
exec figlet -c ANN & 
exec python3.7 ~/breast_cancer_detection/ANN.py &
exec figlet -c KNN &
exec python3.7 ~/breast_cancer_detection/KNN.py &
exec figlet -c SVM & 
exec python3.7 ~/breast_cancer_detection/SVM.py
