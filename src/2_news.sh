#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 1_iris.sh ]
#   Synopsis     [ script that runs Libsvm on the Iris dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#*********************************************************************************************#


#---arguments---#
LIBSVM_PATH=${1:-/home/andi611/LibSVM-Classification/libsvm-3.23}
TRAIN_DATA_PATH=${2:-../data/news/news.tr}
TEST_DATA_PATH=${3:-../data/news/news.te}
OUTPUT_FILE_PATH=${4:-../result/news_predict.csv}


#---variables---#
MODEL_NAME=model_news.libsvm
RUN_BEST=Trues


#---run best---#
if [ "${RUN_BEST}" = True ] ; then
	echo
	echo "|------------------------------------------|"
	echo "|------ Running with Best Parameters ------|"
	echo "|------------------------------------------|"

	$LIBSVM_PATH/svm-train -s 0 -t 0 -q ${TRAIN_DATA_PATH} ${MODEL_NAME}
	echo "Training:"
	$LIBSVM_PATH/svm-predict ${TRAIN_DATA_PATH} ${MODEL_NAME} ${OUTPUT_FILE_PATH}.train
	rm ${OUTPUT_FILE_PATH}.train
	echo "Testing:"
	$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH} ${MODEL_NAME} ${OUTPUT_FILE_PATH}

#---search best---#
else
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

	echo
	echo "|--------------------------------------------------------------|"
	echo "|------ Comparing Different Kernal Functions with Scaling------|"
	echo "|--------------------------------------------------------------|"

	for ((idx=0; idx<${#kernals[@]}; ++idx));
	do
		echo
		echo ">>> Kernal function: ${kernals[idx]}"
		$LIBSVM_PATH/svm-scale -l 0 -u 10 ${TRAIN_DATA_PATH} > ${TRAIN_DATA_PATH}.scale
		$LIBSVM_PATH/svm-scale -l 0 -u 10 ${TEST_DATA_PATH} > ${TEST_DATA_PATH}.scale
		$LIBSVM_PATH/svm-train -s 0 -t ${idx} -q ${TRAIN_DATA_PATH}.scale ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH}.scale ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${TRAIN_DATA_PATH}.scale ${TEST_DATA_PATH}.scale ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done

fi

echo
echo "Done!"

