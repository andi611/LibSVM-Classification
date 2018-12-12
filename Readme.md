# Data Mining: Classification with LIBSVM
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_NEWS.png)
- Datasets:
    - Iris
    - News (subset of 20 Newsgroups dataset, with testing label)
    - Abalone
    - Income (UCI Adult Income dataset)


## Environment
* **< [libsvm](https://github.com/cjlin1/libsvm) >**
* **< numpy 1.15.4 >**
* **< pandas 0.23.4 >**
* **< Python 3.7 >**
* **< tqdm 4.28.1 >** (optional - progress bar)
* **< graphviz 0.10.1 >** (optional - visualization)
 

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
### Setup
- Unzip `data.zip` with: `unzip data.zip`



## Result - Naive Bayes Performance


## Data Preprocessing
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/income_preprocessing.png width="423" height="70">

