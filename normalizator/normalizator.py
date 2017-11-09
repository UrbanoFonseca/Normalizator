import numpy as np
import copy
from sklearn.base import BaseEstimator, TransformerMixin

class StandardScaler(BaseEstimator, TransformerMixin):
	# Z-SCORE NORMALIZATION
	# For each feature vector, apply:
	# v = [v - mean(v)] / std(v)
	def __init__(self):
		# Save the function parameters
		self.means = []
		self.stds = []

	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			v = X_[:, i]
			self.means.insert(i, np.mean(v))
			self.stds.insert(i, np.std(v))

	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] = (X_[:, i] - self.means[i])/self.stds[i]
		self.X_ = X_
		return self.X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_



class MinMaxScaler(BaseEstimator, TransformerMixin):
	# MIN MAX NORMALIZATION
	# For each feature vector, apply:
	# v = [v - min(v)] / [max(v) - min(v)]
	def __init__(self):
		self.min = []
		self.max = []

	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			v = X_[:, i]
			self.min.insert(i, np.min(v))
			self.max.insert(i, np.max(v))

	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] = (X_[:, i] - self.min[i])  / (self.max[i] - self.min[i])
		self.X_ = X_
		return X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_


class DecimalScaler(BaseEstimator, TransformerMixin):
	# DECIMAL SCALING
	# Normalize each feature by dividing each feature sk by 10^n,
	# where n = log10(max(si))
	def __init__(self):
		self.n = []

	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			self.n.insert(i, np.ceil(np.log10(np.max(X_[:, i]))))
		
	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] /= 10**self.n[i]
		self.X_ = X_
		return X_

	def fit_transform(self, X):
		self.fit(X)
		self.transform(X)
		return self.X_


class MedianScaler(BaseEstimator, TransformerMixin):
	# As presented in the paper:
	# Statistical Normalization and Backpropagation for Classification
	# Jayalakshmi, T. and Santhakumaran, A.
	def __init__(self):
		self.median = []

	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			self.median.insert(i, np.median(X_[:, i]))

	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] = X_[:, i] / self.median[i]
		self.X_ = X_
		return X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_


class MMADScaler(BaseEstimator, TransformerMixin):
	# Median and Median Absolute Deviation
	# MAD = median(|sk - median|)
	def __init__(self):
		self.median = []
		self.mad = []
        
	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			self.median.insert(i, np.median(X_[:, i]))
			self.mad.insert(i, np.median(np.abs(X_[:, i] - np.median(X_[:, i]))))
		
	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] = (X_[:, i] - self.median[i]) / self.mad[i]
		self.X_ = X_
		return X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_



    
class MAXScaler(BaseEstimator, TransformerMixin):
	# As presented in the paper:
    # Efficient approach to Normalization of Multimodal Biometric Scores (2011)
    # of L. Latha and S. Thangasamy
    # Min Max Normalization with min = 0.
	def __init__(self):
		self.max = []
        
	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			self.max.insert(i, np.max(X_[:, i]))
		
	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] /= self.max[i]
		self.X_ = X_
		return X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_
    

class modtanhScaler(BaseEstimator, TransformerMixin):
	# As presented in the paper:
	# Efficient approach to Normalization of Multimodal Biometric Scores (2011)
	# of L. Latha and S. Thangasamy
	# tanh s' = 0.5[tanh(0.01(s- mu)/sigma)+1]
	# The np.arctanh is the inverse of the hyperbolic tangent.
	def __init__(self):
		self.mu = []
		self.sigma = []

	def fit(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			self.mu.insert(i, np.mean(X_[:, i]))
			self.sigma.insert(i, np.std(X_[:, i]))

	def transform(self, X):
		X_ = copy.copy(np.asarray(X))
		for i in np.arange(X.shape[1]):
			X_[:, i] = np.arctanh(0.5 * (np.tanh(0.01*(X_[:, i] - self.mu[i])/self.sigma[i])+1))
		self.X_ = X_
		return X_

	def fit_transform(self, X, y='deprecated'):
		self.fit(X)
		self.transform(X)
		return self.X_
