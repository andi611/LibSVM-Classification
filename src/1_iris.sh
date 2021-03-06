#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 1_iris.sh ]
#   Synopsis     [ script that runs Libsvm on the Iris dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#*********************************************************************************************#


#---arguments---#
LIBSVM_PATH=${1:-/home/andi611/LibSVM-Classification/libsvm-3.23}
TRAIN_DATA_PATH=${2:-../data/iris/iris.tr}
TEST_DATA_PATH=${3:-../data/iris/iris.te}
OUTPUT_FILE_PATH=${4:-../result/iris_predict.csv}


#---variables---#
MODEL_NAME=model_iris.libsvm
MODE=RUN_BEST
#ODE=COMPARE_KERNAL
#MODE=COMPARE_SCALE
#MODE=RUN_ALL


if [ "${MODE}" = RUN_BEST ] || [ "${MODE}" = RUN_ALL ] ; then
	echo
	echo "|------------------------------------------|"
	echo "|------ Running with Best Parameters ------|"
	echo "|------------------------------------------|"

	$LIBSVM_PATH/svm-train -s 0 -t 0 ${TRAIN_DATA_PATH} ${MODEL_NAME}
	echo "Training:"
	$LIBSVM_PATH/svm-predict ${TRAIN_DATA_PATH} ${MODEL_NAME} ${OUTPUT_FILE_PATH}.train
	rm ${OUTPUT_FILE_PATH}.train
	echo "Testing:"
	$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH} ${MODEL_NAME} ${OUTPUT_FILE_PATH}
fi

if [ "${MODE}" = COMPARE_KERNAL ] || [ "${MODE}" = RUN_ALL ] ; then
	echo
	echo "|--------------------------------------------------|"
	echo "|------ Comparing Different Kernal Functions ------|"
	echo "|--------------------------------------------------|"

	kernals=( linear polynomial radial_basis sigmoid precomputed_kernal )
	for ((idx=0; idx<${#kernals[@]}; ++idx));
	do
		echo
		echo ">>> Kernal function: ${kernals[idx]}"
		$LIBSVM_PATH/svm-train -s 0 -t ${idx} -q ${TRAIN_DATA_PATH} ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH} ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done
fi

if [ "${MODE}" = COMPARE_SCALE ] || [ "${MODE}" = RUN_ALL ] ; then

	echo
	echo "|---------------------------------------------------------------|"
	echo "|------ Comparing Different Kernal Functions with Scaling ------|"
	echo "|---------------------------------------------------------------|"

	kernals=( linear polynomial radial_basis sigmoid precomputed_kernal )
	for ((idx=0; idx<${#kernals[@]}; ++idx));
	do
		echo
		echo ">>> Kernal function: ${kernals[idx]}"
		$LIBSVM_PATH/svm-scale -l -1 -u 1 ${TRAIN_DATA_PATH} > ${TRAIN_DATA_PATH}.scale
		$LIBSVM_PATH/svm-scale -l -1 -u 1 ${TEST_DATA_PATH} > ${TEST_DATA_PATH}.scale
		$LIBSVM_PATH/svm-train -s 0 -t ${idx} -q ${TRAIN_DATA_PATH}.scale ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH}.scale ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${TRAIN_DATA_PATH}.scale ${TEST_DATA_PATH}.scale ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done
fi

echo
echo "Done!"

