#
# 1. DATA PREPROCESSING
#

# IMPORTING THE LIBRARIES
import numpy as np # allows to work with arrays
import matplotlib.pyplot as plt # the pyplot module of mathplotlib library allows to plot charts
import pandas as pd # allows to import the dataset and create the matrix of features and dependant variables vector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# IMPORTING THE DATASET AND CREATING THE MATRIX OF FEATURES AND DEPENDANT VARIABLE VECTOR 

dataset = pd.read_csv('Data.csv') # It returns a dataframe from the csv file

# In any dataset with which we are going to train a ML model,
# we have always 2 entities: 
#   - the features: the column with which we are going to predict the result values
#   - the dependant variable vector: the values you want to predict (usually, the last column)
# ML models expect these 2 entities as separate inputs
X = dataset.iloc[:, :-1].values # all the rows, all the columns except the last one
y = dataset.iloc[:, -1].values # all the rows, only the last column

print(X)
print(y)

# HANDLING MISSING DATA 
# Replace the missing value by the average of all the columns 
# in which the missing data is.
# 
# scikit-learn: data science library containing a lot of tools for data preprocessing tools
# We use SimpleImputer from scikit-learn
from sklearn.impute import SimpleImputer

# create an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# apply the imputer object to the matrix of features by using the fit method
# fit expect as input all the columns with numerical values only
imputer.fit(X[:, 1:3]) # fit the imputer object to the matrix of features
# transform method with apply the transformation of the missing values
# It will return the transformed array
X[:, 1:3] = imputer.transform(X[:, 1:3]) # transform the matrix of features

print(X)

# ENCODING CATEGORICAL DATA
# Encoding category column data into numbers in order to allow ML to correlate them
# to the other numeric data.
# We want to avoid the model to guess that there is an ordering correlation between the
# categorical data, if this not the case. So we don't assign 0, 1, 2... to them.
# We apply One Hot Encoding: It consists in turning the column data into n columns,
# where n is the number of different categories of data in the column (3 in the example).
# We create binary vectors for each of the categories.
# e.g.: France = [1, 0, 0], Spain = [0, 1, 0], Germany = [0, 0, 1]

# a. Encoding the Independent Variable
# We use: 
#   - ColumnTransformer class from the compose module of the scikit-learn library 
#   - OneHotEncoded class from the preprocessing module of the scikit-learn library
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 

# Create an instance of the ColumnTransformer class
# It will takes 2 arguments:
#    - transformers: a tuple between [] specifying: 
#         * the kind of transformation
#         * the class which implement the transformation 
#         * the index of the column to trasform
#    - remainder: the code name: 
#         * 'passthrough': specify that we want to keep the columns that 
#                          won't be applied some transformation
#       
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# Fitting the transformation
# the output must be transformed in a NumPy array
X = np.array(ct.fit_transform(X))

print(X)

# b. Encoding the Dependent Variable
# Encode the dependent variable (Purchased in the example)
# we call LabelEncoder class
# The dependent variable doesn't need to be a NumPy array
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET 
# train_test_split() function from model_selection module of sklearn library
# provides 4 matrixes: X_train, y_train, X_test, y_test
# which are 2 sets splitted in features and dependant variables
# It takes the X and y, the test size (80/20 recommended)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# FEATURE SCALING
# It is meant to avoid that some features prevails on the others
# It is not need to apply to all the ML models, but just to some of them
# For example, it is not needed for Multiple linear Regression
# The main two technics are:
#   - Standardization: subtracting each value of the features 
#                      by the mean of all the features and then 
#                      dividing by the standard deviation (the
#                      square root of the variants). All the variables
#                      of the features will results in the range [-3, +3].
#                      Good for features following normal (gauassian) distribution.
#   - Normalization: subtracting each value of the features 
#                    by the minimum value of the features, and then
#                    dividing by the difference between the max and
#                    of the features will results in the range [0, 1].
#                    It works well all the times.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# It doesn't apply to the encoded categorical variables
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

# we apply just transform to the test set to have the same transformation
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)



