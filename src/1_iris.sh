#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 1_iris.sh ]
#   Synopsis     [ script that runs Libsvm on the Iris dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#*********************************************************************************************#

LIBSVM_PATH=${1:-/home/andi611/LibSVM-Classification/libsvm-3.23}
TRAIN_DATA_PATH=${2:-../data/iris/iris.tr}
TEST_DATA_PATH=${3:-../data/iris/iris.te}
OUTPUT_FILE_PATH=${4:-../result/iris_predict.csv}

echo "Running different kernal functions..."
kernals=( 1 2 3 4 5 )
for i in "${kernals[@]}"
do
	$LIBSVM_PATH/svm-train -s 0 -t $i -q $TRAIN_DATA_PATH ./model_iris.libsvm
	$LIBSVM_PATH/svm-predict $TEST_DATA_PATH ./model_iris.libsvm $OUTPUT_FILE_PATH
done
