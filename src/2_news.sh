#!/bin/bash
#*********************************************************************************************#
#   FileName     [ 2_news.sh ]
#   Synopsis     [ script that runs Libsvm on the News dataset ]
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
MODE=RUN_BEST
#MODE=COMPARE_KERNAL
#MODE=COMPARE_CSVM
#MODE=COMPARE_SCALE
#MODE=RUN_ALL


#---run best---#
if [ "${MODE}" = RUN_BEST ] || [ "${MODE}" = RUN_ALL ] ; then
	echo
	echo "|------------------------------------------|"
	echo "|------ Running with Best Parameters ------|"
	echo "|------------------------------------------|"

	$LIBSVM_PATH/svm-train -s 0 -t 0 -e 0.01 -w3 2.5 ${TRAIN_DATA_PATH} ${MODEL_NAME}
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
		$LIBSVM_PATH/svm-train -s 1 -t ${idx} -e 0.01 -q ${TRAIN_DATA_PATH} ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH} ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done
fi

if [ "${MODE}" = COMPARE_CSVM ] || [ "${MODE}" = RUN_ALL ] ; then
	echo
	echo "|-------------------------------------------------------|"
	echo "|------ Comparing Different Linear C-SVM Settings ------|"
	echo "|-------------------------------------------------------|"

	epsilons=( 0.00001 0.0001 0.001 0.01 0.1 1.0 2.0 3.0 4.0 10.0 )
	for ((idx=0; idx<${#epsilons[@]}; ++idx));
	do
		echo
		echo ">>> epsilon value: ${epsilons[idx]}"
		$LIBSVM_PATH/svm-train -s 0 -t 0 -e ${epsilons[idx]} -q ${TRAIN_DATA_PATH} ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH} ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done

	classes=( 0 1 2 3 )
	for ((idx=0; idx<${#classes[@]}; ++idx));
	do
		echo
		echo ">>> Class i: ${classes[idx]}"
		$LIBSVM_PATH/svm-train -s 0 -t 0 -e 0.01 -w${classes[idx]} 1.5 -q ${TRAIN_DATA_PATH} ${MODEL_NAME}.temp
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
		$LIBSVM_PATH/svm-scale -l 0 -u 1 ${TRAIN_DATA_PATH} > ${TRAIN_DATA_PATH}.scale
		$LIBSVM_PATH/svm-scale -l 0 -u 1 ${TEST_DATA_PATH} > ${TEST_DATA_PATH}.scale
		$LIBSVM_PATH/svm-train -s 1 -t ${idx} -e 0.01 -q ${TRAIN_DATA_PATH}.scale ${MODEL_NAME}.temp
		$LIBSVM_PATH/svm-predict ${TEST_DATA_PATH}.scale ${MODEL_NAME}.temp ${OUTPUT_FILE_PATH}
		rm ${TRAIN_DATA_PATH}.scale ${TEST_DATA_PATH}.scale ${OUTPUT_FILE_PATH} ${MODEL_NAME}.temp
	done
fi

echo
echo "Done!"

