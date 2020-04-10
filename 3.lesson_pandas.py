"""
Last amended: 7th February, 2019
Myfolder:  /home/ashok/Documents/1.basic_lessons


Reference: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#series
           https://docs.python.org/2/tutorial/introduction.html#unicode-strings


Objectives:
	i)  Data structures in pandas: Series, DataFrame and Index
	ii) Data structures usage


"""

import pandas as pd
import numpy as np
import os


########Series#############
## A. Creating Series
# 10. Series is one-dimensional (columnar) index labeled
#      array. Labels are referred as index. Series may have
#        dtype as float64, int64 or object

# 10.1 Exercises
s = pd.Series([2,4,8,10,55])
s
type(s)
s.name = "AA"
s


# 10.2 This is also a series but stores list objects
t = pd.Series({'a' : [1,2,3,4,], 'b' : [5,6]})
t
type(t)



# 10.3 Exercise
ss=[23,45,56]
h=pd.Series(ss)
h

# 10.4 OR generate it as:
h=pd.Series(range(23,30,2))
h


## B. Simple Operations
# 10.5 Exercise
s+h
s*h
s-h

(s+h)[1]       # Note the indexing starts from 0
s*h[2]


s.mean()
s.std()
s.median()




## C. Series as ndarray
 # 10.6 Also series behaves as ndarray
 #      Series acts very similarly to a ndarray,
 #      and is a valid argument to most NumPy functions.
np.mean(s)
np.median(s)
np.std(s)


## D. Indexing in series
# 10.7 Exercise
d=pd.Series([4,5], index=['a','b'])
e=pd.Series([6,7], index=['f','g'])
f=pd.Series([9,10], index=['a','b'])
d+e  # All NaN
d+f


# 10.8 Reset index of 'd' and check
v = d.reset_index()
type(v)            # v is a DataFrame


# 10.9
d.reset_index(
              drop = True,     # drop = False, adds existing index as
              inplace = True   # a new column and makes it a DataFrame
              )

d

e.reset_index(drop = True, inplace = True)
d + e


## E. Accessing Series
# 10.10 Exercise
j= pd.Series(np.random.normal(size=7))

k=j[j>0]
k=j[j>np.mean(j)]
k


# 10.11 Exercise
k = pd.Series(
             np.random.normal(size=7),
             index=['a','b','c','d','e','f','a']
             )

k['a']   # 'a' is duplicate index
k.loc['a']
k[:2]    # Show first two or upto 2nd index (0 and 1)
k.iloc[:2]

# 10.12
k.iloc[2:]    # Start from 2nd index
k.iloc[2:4]   # Start from IInd index upto 4th index
k.iloc[2:4].mean()


# 10.13  SURPRISE HERE!
k = pd.Series(np.random.normal(size=7),index=[0,2,5,3,4,1,6])
k.loc[0]                 # Access by index-name
k.loc[1]                 # Access by index-name
k.iloc[:2]                # Access by position
k.iloc[[0,1,2]]           # Access by index-name
k.take([0,1,2])           # Access by position
k.loc[[0,1,2]]


# 10.8 Exercise
# A series is like a dictionary. Can be accessed by its index (key)
e=pd.Series(np.random.uniform(0,5,7), index=['a','b','c','d','e','f','g'])
e
e['a' : 'e']
e.loc['a' : 'e']

e['a' : 'd']   # All values from 'a' to 'd'
e['b' : 'd']
e.take(['b' : 'd'])
e+k


######## DataFrame ###########

'''
DataFrame is a 2-dimensional labeled data structure with columns
of potentially different types. You can think of it like a spreadsheet
or SQL table, or a dict of Series objects. It is generally the most
commonly used pandas object. Like Series, DataFrame accepts many
different kinds of input.
'''

# 1
path = "/home/ashok/datasets/delhi_weather"
# 2
os.getcwd()
# 3
os.chdir (path)
# 4
data=pd.read_csv("delhi_weather_data.zip")
type(data)
# 5.1
pd.options.display.max_columns = 200
# 5.2
data.head()
data.tail()
data.dtypes
data.shape
data.columns
data.values
data.columns.values
data.describe()


# 6. Datetime conversions
data['datetime'] = pd.to_datetime(data['datetime_utc'])
data.head()

# 6.1
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['weekday'] = data['datetime'].dt.weekday
data['hour'] = data['datetime'].dt.hour
data['week'] = data['datetime'].dt.week
data.head()

# 6.2
pd.unique(data['_conds'])	# Unique values
data['_conds'].nunique()      # 39
data['_conds'].value_counts().sort_values(ascending = False)
data.head()

# 7.0 Integer Selection
data.columns
data.iloc[3:5, 1:2]  # 3:5 implies start from 3rd pos uptil 5-1=4th pos
data.iloc[3:5, 1:3]  # Display column numbers 2nd and 3rd
data.iloc[3:5, :]		# Display all columns
data.iloc[3:5, :]		# Display all columns
data.iloc[1,1]        # Same as df[1,1:2]. Treat 1 as lower bound
data.iloc[[3,5,7],[1,3]]		# Specific rows and columns
data[data.month == 10 ].head()   # Boolean indexing
data[(data.month == 10) & (data["_conds"] == 'Smoke') ].head()
data[(data._conds == 'Smoke') | (data._wdire == 'East')]


# 8.0 Overall how many values are nulls
np.sum(data.isnull()).sort_values(ascending = False)


# 9.0 Converting categorical variables to numeric
#     sklearn's labelencoder is one way to do it
#     Two step process:
#                1st. Convert dtype from 'object' to 'category'
#                2nd. Get integer-codes behind each category/level
#                3rd. Get correspondence behind category and integer
data['_conds'] = data['_conds'].astype('category')  # Convert to categorical variable
data['int_conds']=data['_conds'].cat.codes          # Create a column of integer coded categories
x = data[['_conds', 'int_conds']].values            # Get dataframe as an array
out = set([tuple(i) for i in x])                    # Get unique tuples of (code,category)



# 10.0 Memory reduction by changing datatypes
data.dtypes

# 10.1 Select data subset with dtype as 'float64'
newdata = data.select_dtypes('float64')

# 10.2 What are max and min data values
np.min(np.min(newdata))        # -9999
np.max(np.max(newdata))        # 101061443.0

# 10.3 What are the limits of various float datatypes
np.finfo('float64')    # finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)
np.finfo('float32')    # finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
np.finfo('float16')    # finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)
np.iinfo('int64')
np.iinfo('int16')

# 10.4 Change all columns to float32
# 10.4.1 What is the present memory usage
np.sum(newdata.memory_usage())             # 8887200
# 10.4.2 Change data type now
for col in newdata.columns.values:
    newdata[col] = newdata[col].astype('float32')
# 10.4.3 What is the current datausage
np.sum(newdata.memory_usage())             # 4443640 (around 50% reduction)

###############################################################################
