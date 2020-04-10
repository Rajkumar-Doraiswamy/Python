# -*- coding: utf-8 -*-
"""
Last amended: 21st Feb, 2019
My folder:    /home/ashok/Documents/5.decisiontree
Data folder:  /home/ashok/datasets/german_credit
Data source: UCI repository
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

Objectives:
    i)    Read and explore data
    ii)   Process data using sklearn
    iii)  Learn to build decision tree model
    iv)   Vary decision tree parameters and check
          how accuracy is affected
    v)    Feature importance

"""

# 1.0 Reset memory
%reset -f
# 1.1 Call libraries
import numpy as np
import pandas as pd
# 1.2 For OS related operations
import os
import matplotlib.pyplot as plt

# 1.3 Class for applying multiple data transformation jobs
from sklearn.compose import ColumnTransformer as ct
# 1.4 Scale numeric data
from sklearn.preprocessing import StandardScaler as ss
# 1.5 One hot encode data--Convert to dummy
from sklearn.preprocessing import OneHotEncoder as ohe
# 1.6 for data splitting
from sklearn.model_selection import train_test_split

# 1.7 Modeler
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# User guide: https://scikit-learn.org/stable/modules/tree.html
from sklearn.tree import DecisionTreeClassifier as dt


###############

# 1.9 Kill warnings
import warnings
warnings.filterwarnings("ignore")


# 2.0 Change your working folder
#     and check files therein
os.chdir("D:\\Raj\\Training\\Big Data\\Python\\decisiontree")
os.listdir()

# 2.1 Change ipython options to display all data columns
pd.options.display.max_columns = 300

# 3.0 Read data from zip file
german = pd.read_csv("german_credit.csv.zip")


# 3.1 Get to know data:
german.shape             # 1000 X 20
german.columns
german.dtypes
german.dtypes.value_counts()   # All are int64


# 3.2 Look at data
german.head()             # Target: creditability. All 1's are at top
german.tail()             #  and 0s at the bottom



###########################

# 3.3 Shuffle data
german = german.sample(frac = 1)
german.tail()


# 4.0 Create some new variables
#     from 'age' and 'credit_amount'

german['age_cat'] = pd.cut(german['age'], 3, labels=["0","1","2"])                      # Equal range cut
german['age_qcat'] = pd.qcut(german['age'], 3, labels=["0","1","2"])                    # Equal freq cut


# 4.1
german['credit_amount_cat'] = pd.cut(german['credit_amount'], 3, labels=["0","1","2"])
german['credit_amount_qcat'] = pd.cut(german['credit_amount'], 3, labels=["0","1","2"])


# 4.2 Separate predictors and target
# 4.3 Popup target
y = german.pop('creditability')
y[:3]                 # Pandas Series


# 4.1 Remaining dataframe only has predictors
#     Create an alias of german
X = german
X is german      # X is same as 'german'


### 4.2
###    We now want to know which of the columns are categorical
###    but disguised as integers


# 4.3 How many unique vales per column.
#     Check every column
#     We will assume that if unique values are 4 or less
#     it is categorical column else numeric
X.nunique()
X.nunique() < 5    # All True are categorical

# 4.3.1 Set difference operation
X.columns
r = ['account_balance', 'previous_credit_payment_status',
     'credit_duration_month',
     'purpose_of_earlier_credit', 'credit_amount']

set(X.columns).difference(set(r))


# 4.4 Define a function to separate out categorical/numerical columns
def sep_Cat_Num_columns(dx):
    """
    Takes as input DataFrame
    Returns two lists:
      i)  One of categorical columns
      ii) Other of remaining numerical columns
    """
    # 4.4.1
    cat = dx.nunique() < 5                         # Will give True/False
    # 4.4.2
    categorical_columns = dx.loc[: , cat].columns  # List of cat columns
    # 4.4.3 Remaining are numeric columns
    numerical_columns = set(dx.columns).difference(set(categorical_columns))
    # 4.4.4 Return a list of both columns
    return list(categorical_columns),list(numerical_columns)


# 5.0 Get the columns now
sep_Cat_Num_columns.__doc__

# 5.1
categorical_columns,numerical_columns = sep_Cat_Num_columns(X)
categorical_columns
numerical_columns



###########################
## Process/standardise data
###########################

#### Data Processing and Modeling: Simple Expt first

# 6.0 Create a small DataFrame with two categorical and two numeric variables
dk = pd.DataFrame({'cat':    ['h', 'h', 'l', 'm', 'l', 'm'],
                   'store' : ['a', 'a', 'b' ,'b', 'a','b'],
                   'price' : [2,3,5,9,10,11],
                   'qty'   : [100,200,400,800,900,900]
                   })
dk

# 8.1 OneHotEncode categorical variables
onehot = ohe(sparse = False)               # Create object instance
onehot.fit(dk[['cat', 'store']])           # Learn data
t = onehot.transform(dk[['cat','store']])  # Now transform
t

# 8.2 Scale numeric data
scaleit = ss()
scaleit.fit(dk[['price','qty']])
scaleit.transform(dk[['price','qty']])


# 8.3 Columnar transformer: Two-in-one
#     Use both ohe and scaler in a composite operation

# 8.4 Define operations to perform and on which columns
#     Format: (name, transformer, columns)

op1 = ("cat_col", ohe(sparse = False), ['cat', 'store'])
op2 = ("num_col", ss(), ['price','qty'])


# 8.5 Create column-transformer object to perform these operations
#     The object contains one list of operations
col_transformer = ct([op1,op2])     # Instaniation

# 8.6 Fit and transform now
col_transformer.fit(dk)             # Learn data (dk)
u = col_transformer.transform(dk)   # Transformation of dk
u

########### Expt finished. Now create function

# 9.0 Following function does all the above
def transform(categorical_columns,numerical_columns,X):
    #  Create a tuple of processing tasks:
    #  (taskName, objectToPerformTask, columns-upon-which-to-perform)
    # 9.1 One hot encode categorical columns
    cat = ('categorical', ohe() , categorical_columns  )
    # 9.2 Scale numerical columns
    num = ('numeric', ss(), numerical_columns)
    # 9.3 Instantiate columnTransformer object to perform task
    #     It transforms X separately by each transformer
    #     and then concatenates results.
    col_trans = ct([cat, num])
    # 9.4 Learn data
    col_trans.fit(X)
    # 9.5 Now transform X
    X_tarnsAndScaled = col_trans.transform(X)
    # 9.6 Return transformed data and also transformation object
    return X_tarnsAndScaled, col_trans


# 10.0 Transform both datasets
X_tarnsAndScaled, _  = transform(categorical_columns, numerical_columns, X)


# 10.1
X_tarnsAndScaled.shape           # 1000 X 61
X_tarnsAndScaled[:5, :3]         # See dummy variables
X_tarnsAndScaled[:5,58:]         # See only numeric columns`



###########################
## Split and model
###########################

# 11 Split into train and test datasets AS ALSO GET INDICIES
X_train,X_test, y_train, y_test = train_test_split(
                                                    X_tarnsAndScaled,    # Predictors
                                                    y,                   # Target
                                                    test_size = 0.3      # split-ratio
                                                    )



# 11.2 Check the splits
X_train.shape       # 700 X 61
X_test.shape        # 300 X 61
y_train.shape       # (700,)
y_test.shape        # (300,)


# 12.0 Modeling
# 12.1 Instantiate decision tree modeling object
clf = dt()            # Accept all default parameters
# 12.2 Training
clf.fit(X_train,y_train)   # Train now
# 12.3 Prediction
out = clf.predict(X_test)
# 12.4 Accuracy?
np.sum(out == y_test)/y_test.values.size



### 13.0 Column importance
# 13 Which columns are important
impt = clf.feature_importances_
cols = categorical_columns + numerical_columns

# 13.1 Sort out zip
sorted(zip(impt,cols), reverse = True)


##########################################

# 14.5 Instantiate modeler class
clf = dt(min_samples_leaf = 5)    # Change number of data-points on leaf
# 14.1 Train and develop model
clf.fit(X_train,y_train)
# 14.2 Predict
out = clf.predict(X_test[cols])
# 14.3 Accuracy
np.sum(out == y_test)/y_test.values.size   # No change



###################### I am done ##############################################
