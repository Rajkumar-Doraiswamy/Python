"""
Last amended: 06/02/2019

Objective: Simple experiments using pandas groupby


By “group by” we are referring to a process involving one or
more of the following steps:
    i)   Splitting the data into groups based on some criteria.
    ii)  Applying a function to each group independently.
    iii) Combining the results into a data structure.

Out of these, the split step is the most straightforward.
In fact, in many situations we may wish to split the data
set into groups and do something with those groups. In the
apply step, we might wish to one of the following:

    i)   Aggregation: compute a summary statistic (or statistics) for each group.
    ii)  Transformation: perform some group-specific computations and return a like-indexed object.
    iii) Filtration: discard some groups, according to a group-wise computation that evaluates True or False.



"""

# 1.0 Call libraries
import pandas as pd
import numpy as np

# 2.0 Define a simple dataframe
#     Specify a list of tuples. Each tuple
#     constitutes a row of dataframe.
#     column (headings) and row-names are to be
#     specified separately.
df = pd.DataFrame([
                     ('bird', 'Falconiformes', 389.0, 21.2),     # row 1
                     ('bird', 'Psittaciformes', 24.0, 23.5),     # row 2
                     ('mammal', 'Carnivora', 80.2, 29.0),        # row 3
                     ('mammal', 'Primates', np.nan, 30.6),       # row 4
                     ('mammal', 'Carnivora', 58, 40.8),
                     ('fish', 'Whale', 89, 120.8),
                     ('fish', 'Shark', 78, 80.8)
                  ],
                  index =   ['falcon', 'parrot', 'lion', 'monkey', 'leopard','whale','shark'],
                  columns = ('class', 'order', 'max_speed', 'max_wt'))

df

#############
## 3. Splitting
#############
# Various ways to groupby
#  default grouping is by is axis=0

# Collectively we refer to the grouping objects as the keys.
grouped = df.groupby('class')      # Same as: df.groupby(['class'])
grouped1 = df.groupby(['class', 'order'])

grouped      # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e2128>
grouped1     # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e27b8>



###########################
##4. GroupBy object attributes
###########################
grouped.groups           # Describes groups dictionary
grouped1.groups          # Dict object
len(grouped)             # How many items are there in the group
len(grouped1)            # 6


# 4.1 Iterating through the groups
#     Peeping into each basket
for name, group in grouped:
    print(name)
    print(group)

# 4.2 Out of these multiple boxes/groups
#     A single group can be selected using get_group():
grouped.get_group('fish')


##############
## 5. Aggregating
#############
# Once the GroupBy object has been created,
# several methods are available to perform
# a computation on each one of the groups.

# 5.1
grouped['max_speed'].sum()     # keys are sorted
# OR
grouped.max_speed.sum()

"""
# Summary functions are
mean() 	   Compute mean of groups
sum() 	   Compute sum of group values
size() 	   Compute group sizes
count()    Compute count of group
std() 	   Standard deviation of groups
var() 	   Compute variance of groups
sem() 	   Standard error of the mean of groups
describe() Generates descriptive statistics
first()    Compute first of group values
last() 	   Compute last of group values
nth() 	   Take nth value, or a subset if n is a list
min() 	   Compute min of group values
max()      Compute max of group values
"""

# 5,2 With grouped Series you can also pass a
# list or dict of functions to do
# aggregation with, outputting a DataFram

grouped['max_speed'].agg([np.sum, np.mean, np.std])

# 5.3 By passing a dict to aggregate you can apply a
#     different aggregation to the columns of a DataFrame:

grouped.agg({'max_speed': np.sum,
             'max_wt': np.std })


##############
## 6. Class Exercises:
#############

%reset -f
import pandas as pd
import numpy as np

# 6.0 Create dataframe
# 6.1 First create a dictionary
dd = {'age' :  np.random.uniform(20,30,10), 'city' : ('del', 'fbd') * 5}
dd

# 6.1 Next dataframe
abc = pd.DataFrame(dd)
abc

# 7.0 Now answer these questions
# Q 7.1. Group by city and show groups:
grouped = abc.groupby(['city'])
grouped.groups

# Q 7.2. Show minimum of age in each group
grouped['age'].min()

# Q 7.3. Just get 'del' group
grouped.get_group('del')

# 8. Change the above dataframe as follows:

dd['gender'] = list('mmmmmmfffm')
dd['income'] = np.random.random(10)
dd
cde = pd.DataFrame(dd)
cde

# Q 8.1: Group by city and gender
grouped1 = cde.groupby(['city','gender'])

# Q 8.2 Find average age by by city and gender
#       Note multiple-indexes
grouped1['age'].mean()

# Q 8.3 Transform one of the indexes as columns
grouped1['age'].mean().unstack()

# Q 8.4. Find average 'age' but 'min' income by 'city' and 'gender'
grouped1['age','income'].aggregate({'age' : 'mean' , 'income': 'min'  })
grouped1['age','income'].aggregate({'age' : 'mean' , 'income': 'min'  }).unstack()


# Q 8.5. Apply multiple functions on each numerical column:
grouped1['age','income'].aggregate({'age' : ['mean', 'max'], 'income' : np.min})

# Q 8.6. Design your own summary function and apply
def wax(ds):
    return((np.sum(ds))**2 )

# Q 8.7. Does the function work?
wax(cde['income'])

# Q 8.8. Now use it on grouped1
grouped1['income', 'age'].agg({'age': [wax, np.sum]})

# Q 8.8.1 Rename columns
grouped1['income', 'age'].agg({'age': [wax, np.sum]}).rename(columns= {'wax': 'sum *2', 'sum': 'summation'})

# Q9. Using apply()
#     Function invoked by 'apply()' takes a dataframe as
#     argument and returns a scalar or a pandas object.
#     In between what you do in the function is your business
grouped1['income', 'age'].apply(wax)
grouped1['income', 'age'].apply(lambda r: np.sqrt(r))


# Q10. Using transform() function: Feature creation
grouped1['income', 'age'].transform(wax)

# Q10 Group by 'city' and summarise gender
