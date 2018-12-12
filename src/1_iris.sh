#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 1_iris.sh ]
#   Synopsis     [ script that runs Libsvm on the Iris dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#*********************************************************************************************#

LIBSVM_PATH=${1:-/home/andi611/dm/libsvm-3.23}
TRAIN_DATA_PATH=${2:-../data/iris/iris.tr}
TEST_DATA_PATH=${3:-../data/iris/iris.te}
OUTPUT_FILE_PATH=${4:-../result/iris_predict.csv}

.$LIBSVM_PATH/svm-train $TRAIN_DATA_PATH ./model_iris.libsvm
.$LIBSVM_PATH/svm-test $TEST_DATA_PATH ./model_iris.libsvm $OUTPUT_FILE_PATH
