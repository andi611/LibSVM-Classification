#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 1_iris.sh ]
#   Synopsis     [ script that runs Libsvm on the Iris dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#*********************************************************************************************#

LIBSVM_PATH ?= /home/andi611/dm/libsvm-3.23
./$(LIBSVM_PATH)/svm-train
