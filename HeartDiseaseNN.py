# # cd tutorial
# # jupyter lab

# 303 instances

# NOTE:

# heart-disease.names

# processed.cleveland.data

# Note: 7 most common 

##########################################

# Begin code:

import sys
import pandas
import numpy as np
import sklearn
import matplotlib
import keras

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Keras: {}'.format(keras.__version__))

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# importing the dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The names for each column  in our pandas DataFrame

#1. #3 (age) 
#2. #4 (sex) 
#3. #9 (cp) 
#4. #10 (trestbps) 
#5. #12 (chol) 
#6. #16 (fbs) 
#7. #19 (restecg) 
#8. #32 (thalach) 
#9. #38 (exang) 
#10. #40 (oldpeak) 
#11. #41 (slope) 
#12. #44 (ca) 
#13. #51 (thal) 
#14. #58 (num) (the predicted attribute) 


names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']

# read the csv

cleveland = pd.read_csv(url, names=names)

# print the shape of the DataFrame, plus some examples

print('Shape of DataFrame: {}'.format(cleveland.shape))
print(cleveland.loc[1])

# Example: (303, 14)

# 303 patients, with the given 14 attributes: age, sex, chest p...

# first: 67 yrs old, male, chest pain, heart disease with class 2. Does not have severe, but still does have heart disease.

# print the last twenty or so data points: from 280 to 303

# Columns with all 14 attributes:

# NOTE: Missing values exist. MUST DO PREPROCESSING. A particular patient may or may not have done a test. Or, a patient may have done a test that no other patients have.



cleveland.loc[280:]

# Remove missing data (indicated with a "?")

data = cleveland[-cleveland.isin(['?'])]

data.loc[280:]

# Goes from ? to NaN as null values. nothing exists in those locations.

###WANT: To drop rows with NaN values from DataFrame.

data = data.dropna(axis = 0)  # axis = 0 -> drop row ; axis = 1 -> drop column
data.loc[280:]

# print the shape and data type of the DataFrame

print (data.shape)
print (data.dtypes)

# (297, 14) -> Here, we dropped 6 rows that included missing values.

# age float64

# ...

# ca object

# class int64

# dtype: object

# Pandas still has columns labeled as objects:

###Want:

# transform data to numeric to enable further analysis.

data = data.apply(pd.to_numeric)
data = data.types

# OUT -> all float64 except class

# print data characteristics, using pandas describe() function

data.describe()

# 297 of each attribute.
# mean, std, min, quartiles, and max

### FIRST, want to do data exploration (EXPLORATORY DATA VISUALIZATION) before Machine Learning techniques!!!   ###

# All numerical data -> Regression Models.

# All categorical data -> Classification Techniques.

### So in this case, we can use classification with Neural Networks.

# In some cases, we want to first normalize our data.

# Note: if we have categorical labels:
# Ex) Chest pain type:
#### Value 1: typical angina
#### Value 2: atypical angina
#### Value 3: non-anginal pain
#### Value 4: asymptomatic

# Vs.

# # 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
### This is a continous, numerical value!!! ###
# can't normalize.

###Why Data Normalization is necessary for Machine Learning models. ###

# Normalization:
# - is a technique often applied as part of data preparation for machine learning. 

# The goal of normalization:
# - To change the values of numeric columns in 
# the dataset to a common scale, w/o distorting 
# differences in the ranges of values.

# EX: 
# 1-10 -> 1
# 11-20 -> 2
# 21-30 -> 3

# plot histogram for each variable

data.hist(figsize = (12,12))
plt.show()

# Reading the histogram:

# Age: 
# - Numerical data.
# - None/barely any from 0-20.
# Starts to increase from 20-40 to 10 instances.
# At its peak at age 60.
# Starts to decrease after age 60 -> 
# Meaning population has already died off. Most who have heart disease at 60, implies population cut off. If the population has been cut off, they won't live until 80. Most people die before they reach 100. It can be by heart disease or other health factors.

# Sex:
# - Categorical data.
# - 100 are 0 (MALE)
# - 200 are 1 (FEMALE)

###IMPLIES: FEMALES make up most of the population. ###

# Over 150 people do not have heart disease.
# Decreasing levels of population that have increasely more severe cases of heart disease.

# NOTE: In ML, the less we have in one particular class, will be reflected in our algorithm. So, 4's will be most uncommon and less likely to predict.

# Will look for ways to correct imbalance.

# Have a good split, however with people who do not and those who do have heart disease.

# RECAP:

# - Preprocessed our data
# - Explored our data
# - imported all our data

#########NEXT: Will use train_test_split. 

# from sklearn.model_selection import train_test_split


# Of Total Data:

# 80% - Training Dataset

# 20% - Testing Dataset

####### THEN, will convert categorical data into class labels. ######

# Create X and Y datasets for training.

from sklearn import model_selection

# WANT: Convert dataframe into a numpy array.

X = np.array(data.drop(['class'],1))  #Take everything from our dataset but the class attribute.

# Want to predict y given X and see how close the values measure/approach y.

y = np.array(data['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_spli(X, y, test_size = 0.2)

# Convert data to categorical labels

from keras.utils.np_utils import to_categorical

Y_train = to.categorical(y_train, num_class=None)

Y_test = to.categorical(y_test, num_classes=None)

# (237L, 5L)

#[[1,0,0,0,0]
#[0,0,1,0,0,]
#...
#...
#...
#]]

# HERE, we have categoical labels: [1,0,0,0,0] -> Category 1, etc.

from keras.models import Sequential
from keras.models import Dense
from keras.optimizers import Adam

# define a function to build the keras mode

def create_model():
  model = Sequential()
  model.add(Dense(8, input_dim=13, kernal_initializer='normal', activation='relu'))  # Dense, fully connected layer w/ 8 neurons, 13 attributes (removed class)
  model.add(Dense(4, kernel_initializer='normal', activation='relu'))
  model.add(Dense(5, activation='softmax')) # GOOD for categorical data

  # Compile the model
  adam = Adam(lr=0.001) # Optimizer
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model

model = create_model()
print(model.summary[])

# Neural Network w/ 3 Dense Layers with given shapes/parameters:

# fit the model to the training data

model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)

# Looks at 10 patients, the gradient, how to update the parameters
# Batch size - number of instances to look at before compiling and calculating parameters.
# will update every 10 patients.

# Verbose = 1 -> Print out information so we can see how training is going.

# Prints out loss and accuracy. 

# Hits a local minimum.

# Is within 0.59 and 0.61 accuracy...

# Want to say what kind and how severe the heart disease is

# WHAT WE WANT: To improve using Binary Classification. We were working with Categorical Classification.

# Convert to binary classification problem - heart disease or no heart disease
Y_train_binary = y_train.copy() # Make copy instead of redefining. Can go back to categorical model afterwards.
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1  
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])
print(Y_test_binary[:20])

# Define a new keras model for binary classification:

def create_binary_model():
  model = Sequential()
  model.add(Dense(8, input_dim=13, kernal_initializer='normal', activation='relu'))  # INPUT LAYER: Dense, fully connected layer w/ 8 neurons, 13 attributes (removed class) 
  model.add(Dense(4, kernel_initializer='normal', activation='relu')) # Hidden Layer
  model.add(Dense(1, activation='sigmoid')) # GOOD for BINARY -> Sigmoid Function. Only one is in this output model: [0 or 1] vs. [0 1 2 3 4] for classification

  # Compile the model
  adam = Adam(lr=0.001) # Optimizer
  model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model

binary_model = create_binary_model
print(binary_model.summary())

# Fit the binary model on the training data

binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)

# 84/85% more accurate.

# WANT TO DO PERFORMANCE METRICS:
# - to see what our results actually mean/what are we actually predicting in each of these cases.

# Generate classification report using predictions for categorical model

from sklearn.metrics import classifcation_report, accuracy_score

categorical_pred = model.predict(X_test) # Predict using test data (X_test)

categorical_pred

# array([6.741e-01,...,...])
# Gives probability for each individual class of heart disease.

# WANT TO CHANGE to binary:
# Make those that are contain a 1, and w/o as a 0.



categorical_pred = np.argmax(model.predict(X_test), axis=1) # Arg - WHichever class has the highest probability, label as 1, else 0

categorical_pred


# array([0, 0,3,3,0,0,0,0,3..])

# NOTE: we are not predicting each and every single level of coronary heart disease

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

# Results:
# 0.666

# Precision, Recall, f1-score, support

# Class 0: Healthy -> Precision: 0.83 -> few false positives (0.17) ;
# Recall: 0.97 -> maybe one false negative (0.03)

# False pos: incorrectly predicts a particular condition exists, when in reality it does exist.
# False neg: incorrectly predicting a particular condition does not exist, when in reality it does exist

# F1 : combo precision/recall

# cannot use this 

# NOW, generate classification report using predictions for the binary model.

binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))

# Results for Binary Model:
# 0.7833333

#         Precision Recall F1-score Support
#   0          0.88    0.77     0.82     39
#   1          0.65    0.81     0.72     21

# Avg/Total    0.80    0.78     0.79      0

# For healthy patients, 88% Precision/ 77% Recall for 39 patients.
# Those with heart disease, we have the following:
# 65% Precision, 81% Recall for 21 patients.

# WITH HEART DISEASE:
# Few false positive, not that many false negatives.
# Few incorrectly predicted to have heart disease, when actually do not.
# Not that many are predicted to not have heart disease, when in reality they do. 

########THIS IS GOOD FOR PATIENTS!!! Better to have more false positives (falsely predicted to have heart disease, when they do not), rather than having more false negatives (predicted they do not have heart disease, when in reality they do)!!!


# MORE USEFUL!!!

### RECAP:

# We will use this information to scan health data of an incoming patient, and the physician will get: 
### This patient has a high chance of heart disease.

#  We predicted with almost 80% accuracy using just 14 attributes. Could've gone by others, such as how often do they smoke?

####### SUMMARY:

#1) how to use sklearn/keras
#2) how to import data from the pd.read_csv(url, names=names)
#3) how to preprocess data by removing rows with missing attributes
#4) describing data
#5) printing out data so we know what we are working with
#6) Using train_test_split and converting one hot encoded vectors for categorical classification
#7) defining our neural networks using keras
#8) types of activation functions: softmax vs. sigmoid
#9) categorical cross-entropy vs. binary cross_entropy
#10) Looked at training data, how do we fit that model to that training data.
#11) How to look at accuracy_score and classification_report: Do we have a lot of false positives and false negatives?
































































































































































































Y_train_binary[Y_train_binary > 0] = 1  
Y_test_binary[Y_test_binary > 0] = 1





































































































































































































































































































































































































































































































































































































































































































