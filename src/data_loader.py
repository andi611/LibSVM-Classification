# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ data_loader.py ]
#   Synopsis     [ Loader that parse the Abalone dataset for LibSVM ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import csv
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='data_loader_args')
	parser.add_argument('--verbose', action='store_true', help='Print data parsing information')

	data_args = parser.add_argument_group('data')
	data_args.add_argument('--data_abalone', action='store_true', help='Training and testing on the Abalone dataset')
	data_args.add_argument('--data_income', action='store_true', help='Training and testing on the Income dataset')

	path_args = parser.add_argument_group('train_path')
	path_args.add_argument('--train_path_abalone', type=str, default='../data/abalone/abalone_train.csv', help='path to the Abalone training dataset')
	path_args.add_argument('--train_path_income', type=str, default='../data/income/income_train.csv', help='path to the Income training dataset')

	path_args = parser.add_argument_group('test_path')
	path_args.add_argument('--test_path_abalone', type=str, default='../data/abalone/abalone_test.csv', help='path to the Abalone testing dataset')
	path_args.add_argument('--test_path_income', type=str, default='../data/income/income_test.csv', help='path to the Income testing dataset')
	
	path_args = parser.add_argument_group('output_path')
	path_args.add_argument('--output_train_path_abalone', type=str, default='./abalone.tr', help='path to save processed abalone training data')
	path_args.add_argument('--output_test_path_abalone', type=str, default='./abalone.te', help='path to save processed abalone testing data')
	path_args.add_argument('--output_train_path_income', type=str, default='./income.tr', help='path to save processed income training data')
	path_args.add_argument('--output_test_path_income', type=str, default='./income.te', help='path to save processed income testing data')

	args = parser.parse_args()
	return args


###############
# DATA LOADER #
###############
class data_loader(object):

	def __init__(self, args):
		
		self.verbose = args.verbose

		#---training paths---#
		self.train_path_abalone = args.train_path_abalone
		self.train_path_income = args.train_path_income
		
		#---testing paths---#
		self.test_path_abalone = args.test_path_abalone
		self.test_path_income = args.test_path_income


	def _read_data(self, path, dtype, skip_header=False, with_label=True):
		data = []
		with open(path, 'r', encoding='utf-8') as f:
			file = csv.reader(f, delimiter=',', quotechar='\r')
			if skip_header: next(file, None)  # skip the headers
			for row in file:
				if dtype == 'float': data.append([float(item) for item in row])
				if dtype == 'str': data.append([str(item).strip() for item in row])
		if with_label:
			data = np.asarray(data)
			return list(data[:, :-1]), list(data[:, -1])
		else:
			return data


	def _check_and_display(self, train_x, train_y, test_x, test_y=[]):
		print('>> [Data Loader] First training sample: ', train_x[0])
		print('>> [Data Loader] Training x data shape:', np.shape(train_x))
		print('>> [Data Loader] Training y data shape:', np.shape(train_y))
		print('>> [Data Loader] Testing x data shape:', np.shape(test_x))
		if len(test_y) != 0: print('>> [Data Loader] Testing y data shape:', np.shape(test_y))
		if len(test_y) != 0: assert np.shape(test_x)[0] == np.shape(test_y)[0]
		assert np.shape(train_x)[0] == np.shape(train_y)[0]


	def _to_different_dtype(self, data, to_float=False):
		new_data = []
		for i, row in enumerate(data):
			new_row = []
			for j, item in enumerate(row):
				if to_float:
					try: new_row.append(float(item)) # dtype == int
					except: new_row.append(str(item)) # dtype == str
				else:
					try: new_row.append(int(item)) # dtype == int
					except: new_row.append(str(item)) # dtype == str
			new_data.append(new_row)
		return new_data


	def _preprocess_abalone(self, train_x, test_x):

		#---separate str and int dtype---#
		train_x = self._to_different_dtype(train_x, to_float=True)
		test_x = self._to_different_dtype(test_x, to_float=True)
		
		#---process string label into one hot---#
		train_x = pd.DataFrame(train_x)
		test_x = pd.DataFrame(test_x)
		for label in "MFI":
			train_x[label] = np.asarray(train_x)[:,0] == label
			test_x[label] = np.asarray(test_x)[:,0] == label

		#---remove the first original string column---#
		return np.asarray(train_x)[:,1:].astype(np.float), np.asarray(test_x)[:,1:].astype(np.float)


	def _preprocess_income(self, train_x, test_x, to_index=False, one_hot=False, norm=False):

		#---separate str and int dtype---#
		train_x = self._to_different_dtype(train_x)
		test_x = self._to_different_dtype(test_x)

		#---replace ? with np.nan---#
		train_x = pd.DataFrame([[np.nan if item == '?' else item for item in row] for row in train_x])
		test_x = pd.DataFrame([[np.nan if item == '?' else item for item in row] for row in test_x])

		#---impute missing value---#
		imputer = DataImputer()
		imputer.fit(train_x)
		train_x = imputer.transform(train_x).values
		test_x = imputer.transform(test_x).values
		
		#---split into categorical and continuous---#
		categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
		continuous_features = [0, 2, 10, 11, 12] # -> drop column 4 educatuin num
		train_x_cat = np.take(train_x, indices=categorical_features, axis=1)
		train_x_con = np.take(train_x, indices=continuous_features, axis=1).astype(np.float64)
		test_x_cat = np.take(test_x, indices=categorical_features, axis=1)
		test_x_con = np.take(test_x, indices=continuous_features, axis=1).astype(np.float64)

		#---transform categocial to index---#
		if to_index:
			for i in range(len(train_x_cat[0])):
				labeler = LabelEncoder()
				labeler.fit(train_x_cat[:,i])
				train_x_cat[:,i] = labeler.transform(train_x_cat[:,i])
				test_x_cat[:,i] = labeler.transform(test_x_cat[:,i])

		#---transform categocial to one hot---#
		if one_hot:
			encoder = OneHotEncoder(handle_unknown='ignore')
			encoder.fit(train_x_cat)
			train_x_cat = encoder.transform(train_x_cat).toarray()
			test_x_cat = encoder.transform(test_x_cat).toarray()

		#---concatenate and split---#
		train_x = np.concatenate((train_x_cat, train_x_con), axis=1)
		test_x = np.concatenate((test_x_cat, test_x_con), axis=1)
		
		# #---normalize continuous data---#
		if norm:
			normalizer = StandardScaler()
			normalizer.fit(train_x)
			train_x = normalizer.transform(train_x)
			test_x = normalizer.transform(test_x)

		return train_x, test_x


	def fetch_abalone(self):
		print('>> [Data Loader] Reading the Abalone dataset...')
		train_x, train_y = self._read_data(self.train_path_abalone, dtype='str')
		test_x, test_y = self._read_data(self.test_path_abalone, dtype='str')
		train_x, test_x = self._preprocess_abalone(train_x, test_x)
		if self.verbose: self._check_and_display(train_x, train_y, test_x, test_y)
		return train_x, train_y, test_x, test_y


	def fetch_income(self):
		print('>> [Data Loader] Reading the Income dataset...')
		train_x, train_y = self._read_data(self.train_path_income, dtype='str')
		test_x = self._read_data(self.test_path_income, dtype='str', with_label=False)
		train_x, test_x = self._preprocess_income(train_x, test_x, to_index=True, one_hot=True, norm=True)
		if self.verbose: self._check_and_display(train_x, train_y, test_x)
		return train_x, train_y, test_x, None


################
# DATA IMPUTER #
################
class DataImputer(TransformerMixin):

	def __init__(self):
		"""
		Impute missing values.
		- Columns of dtype object are imputed with the most frequent value in column.
		- Columns of other types are imputed with mean of column.
		- Reference: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
		"""
	def fit(self, X, y=None):
		self.fill = pd.Series([X[c].value_counts().index[0]
			if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
			index=X.columns)
		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)


####################
# WRITE FOR LIBSVM #
####################
def write_for_LibSVM(file_path, x_data, y_data):
	with open(file_path, 'w') as file:
		for i in range(len(x_data)):
			if y_data != None: file.write(str(y_data[i]))
			else: file.write(str(0))
			line = ''
			for j in range(len(x_data[i])):
				line = line + ' ' + str(j+1) + ':' + str(x_data[i][j])
			file.write(line)
			file.write('\n')


########
# MAIN #
########
"""
    main function
"""
def main():
	
	args = get_config()
	loader = data_loader(args)

	#---fetch data---#
	if args.data_abalone:
		train_x, train_y, test_x, test_y = loader.fetch_abalone()
		write_for_LibSVM(args.output_train_path_abalone, train_x, train_y)
		write_for_LibSVM(args.output_test_path_abalone, test_x, test_y)

	elif args.data_income:
		train_x, train_y, test_x, test_y = loader.fetch_income()
		write_for_LibSVM(args.output_train_path_income, train_x, train_y)
		write_for_LibSVM(args.output_test_path_income, test_x, test_y)


if __name__ == '__main__':
	main()


