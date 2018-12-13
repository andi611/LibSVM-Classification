# Data Mining: Classification with LIBSVM
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_NEWS.png)
- Datasets:
    - Iris
    - News (subset of 20 Newsgroups dataset, with testing label)
    - Abalone
    - Income (UCI Adult Income dataset)


## Environment
* **< [libsvm 3.23](https://github.com/cjlin1/libsvm) >**
* **< scikit-learn 0.20.1 >** (For date preprocessing)
* **< numpy 1.15.4 >**
* **< pandas 0.23.4 >**
* **< Python 3.7 >**
 

## File Description
```
.
├── src/
|   ├── classifiers.py ----------> Implementation of the naive bayes and decision tree classifiers
|   ├── data_loader.py ----------> Data loader that handles the reading and preprocessing of all 3 datasets
|   └── runner.py ---------------> Runner that runs all modes: train + evaluate, search optimal model, visualize model, etc.
├── data/ -----------------------> unzip data.zip
|   ├── income
|   |   ├── income_test.csv
|   |   ├── income_train.csv
|   |   ├── income.names
|   |   └── sample_output.csv
|   ├── mushroom
|   |   ├── mushroom_test.csv
|   |   ├── mushroom_train.csv
|   |   ├── mushroom.names
|   |   └── sample_output.csv
|   └── news
|       ├── news_test.csv
|       ├── news_train.csv
|       └── sample_output.csv
├── image/ ----------------------> visualization and program output screen shots
├── result/ ---------------------> model prediction output
├── problem_description.pdf -----> Work spec
└── Readme.md -------------------> This file
```


## Usage
### Compile LibSVM from binary:
- Use the `libsvm-3.23/` provided in this repo, or compile by yourself: 
- Unzip `libsvm-3.23.zip` with: `$ libsvm-3.23.zip`
- In `libsvm-3.23/` type: `$ make`

### Run LibSVM on the Iris Dataset:
- `$ ./1_iris.sh`
- There are four modes that can be set manually in the script (Line 19-22):
	- `RUN_BEST`: Run training and testing using the best parameter.
	- `COMPARE_KERNAL`: Run training and testing with different kernal settings and compare performance.
	- `COMPARE_SCALE`: Run training and testing with different kernal settings in addition to data scaling and compare performance.
	- `RUN_ALL`: Run everything above.

### Run LibSVM on the News Dataset:
- `$ ./2_news.sh`
- There are four modes that can be set manually in the script (Line 19-23):
	- `RUN_BEST`: Run training and testing using the best parameter.
	- `COMPARE_KERNAL`: Run training and testing with different kernal settings and compare performance.
	- `COMPARE_CSVM`: Run training and testing with C-SVM settings and compare performance.
	- `COMPARE_SCALE`: Run training and testing with different kernal settings in addition to data scaling and compare performance.
	- `RUN_ALL`: Run everything above.

### Run LibSVM on the Abalone Dataset:
- `$ ./3_abalone.sh`
- There are four modes that can be set manually in the script (Line 21-25):
	- `RUN_BEST`: Run training and testing using the best parameter.
	- `COMPARE_KERNAL`: Run training and testing with different kernal settings and compare performance.
	- `COMPARE_CSVM`: Run training and testing with C-SVM settings and compare performance.
	- `COMPARE_SCALE`: Run training and testing with different kernal settings in addition to data scaling and compare performance.
	- `RUN_ALL`: Run everything above.


## Result - Naive Bayes Performance

### Iris Dataset Results
- Best model training accuracy: **98.66%**
- Best model testing accuracy: **100%**
- Parameter and setting for best model: `svm-train -s 0 -t 0`

| Kernal Type  | Testing Accuracy | Testing Accuracy with Scaling |
| ------------- | ------------- | ------------- |
| Linear | **100.00%** | 97.33% |
| Polynomial | 98.66% | 70.66% |
| Radial Basis Function | 97.33% | 98.66% |
| Sigmoid  | 33.33% | 96.00% |


### News Dataset Results
- Best model training accuracy: **97.63%**
- Best model testing accuracy: **84.55%**
- Parameter and setting for best model: `svm-train -s 0 -t 0 -e 0.01 -w3 1.5`

| Kernal Type  | Testing Accuracy | Testing Accuracy with Scaling |
| ------------- | ------------- | ------------- |
| Linear | 83.36% | 79.86% |
| Polynomial | 49.51% | 35.59% |
| Radial Basis Function | 70.91% | 69.02% |
| Sigmoid  | 70.91% | 67.62% |

### Abalone Dataset Results
- Best model training accuracy: **65.16%**
- Best model testing accuracy: **66.63%**

| Kernal Type  | Testing Accuracy | Testing Accuracy with Scaling |
| ------------- | ------------- | ------------- |
| Linear | 66.63% | 57.81% |
| Polynomial | 61.84% | 57.91% |
| Radial Basis Function | 66.25% | 55.90% |
| Sigmoid  | 56.28% | 54.94% |

### Income Dataset Results
- Best model training accuracy: **97.63%**
- Best model testing accuracy: **84.55%**

| Kernal Type  | Testing Accuracy | Testing Accuracy with Scaling |
| ------------- | ------------- | ------------- |
| Linear | % | % |
| Polynomial | % | % |
| Radial Basis Function | % | % |
| Sigmoid  | % | % |

## Data Preprocessing

### Iris Dataset Preprocessing
- None, this dataset is LibSVM ready.

### News Dataset Preprocessing
- None, this dataset is LibSVM ready.

### Abalone Dataset Preprocessing
- Specify each entry to either one of the data type: (`int`, `str`)
- Change the first column (which is categorical and in `str`) into one-hot encoding vectors.
- Write to LibSVM format.

### Income Dataset Preprocessing
- Specify each entry to either one of the data type: (`int`, `str`)
- Identify all missing entries `'?'` and replace them with `np.nan`
- Impute and estimate all missing entries:
    - If dtype is `int`: impute with mean value of the feature column
    - If dtype is `str`: impute with most frequent item in the feature column
- Split data into categorical and continuous and process them separately:
    - categorical features index = [1, 3, 5, 6, 7, 8, 9, 13]
    - continuous features index = [0, 2, 4, 10, 11, 12]
- For categorical data:
    - 8 categorical attributes are transformed into a 99 dimension one-hot feature vector
- For continuous data:
    - Normalize with maximum norm of that feature column
- Re-concatenate categorical features and continuous features.
- Write to LibSVM format.


<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/income_preprocessing.png width="423" height="70">