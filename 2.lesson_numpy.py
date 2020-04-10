# -*- coding: utf-8 -*-
"""
Last amended: 1st February, 2019
Myfolder: C:\Users\ashokharnal\OneDrive\Documents\python
	  /home/ashok/Documents/1.basic_lessons

Ref:
1. https://docs.scipy.org/doc/numpy-dev/user/quickstart.html#
2. https://docs.scipy.org/doc/numpy-dev/user/quickstart.html#
3. http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb
4. http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb



Objectives:
    a. Numpy arrays
    b. Creating pre-defined arrays
    c. Random numbers
    d. Generate multidimensional arrays
    e. Copying arrays
    f. Accessing and Slicing arrays
    g. Stacking and splitting arrays
    h. Vectorized operations on arrays
    i. Aggregating arrays
    j. Comparisons, Masks, and Boolean Logic
    k. Fancy Indexing and sorting arrays
    l. Array looping
    m. Writing your modules and importing them
    n. What is a python package?


"""

'''
Axis in numpy/pandas

+------------+---------+--------+
|            |  A      |  B     |
+------------+---------+---------
|      0     | 0.626386| 1.52325|----axis=1-----> (along rows)
+------------+---------+--------+
                |         |
                | axis=0  |
                ↓ vertically ↓

'''

import numpy as np
np.__version__

# 1. Do not truncate output on screen
#    Default is 1000
np.set_printoptions(threshold=np.inf)

######################### AA. #######################################

# A bit about range in python
# 2. range(start, stop,step)
#     Note the following:

f=range(20, 30, 2)  # all elements MUST be integers. Try range(20,30.,2)
                    # Generally used for indexing
type(f)
list(f)  # Becomes list

# 2.1
g = range(20,10, -3)  # Negative step
list(g)


# 2.2 Or as
k=list(range(10))
k
type(k)
type(k[0])


######################### BB. #######################################
######### 3. Arrays in numpy ##########


# 3.1
np.array([1,2,3])               # shape (3,)   1D array of 3-elements
np.array([[1,2,3]])             # shape (1,3)  2D array of 3-entries in one row
                                #              See below for (3,1)

np.array([[1],[2],[3]])         # Three objects having one element each,
                                #  shape (3,1)  Two dimensional

np.array([[1,2,3],[4,5,6]])     # Two objects, having three elements each: shape (2,3)
np.array([[[1,2,3],[4,5,6]],[[8,9,0],[3,4,5]]])   # shape (2,2,3)
                                                  # 2-outer objects (lists),
                                                  #  within each 2 objects (lists)
                                                  # and within list, each 3 elements


## 3.2 Create an array from list or tuple of python
np.array(list(range(10)))      # shape (10,1)
np.array(range(10)).shape

# 3.3 Multidimensional array, is a list of lists
#     Following works:
np.array(list(range(10)))

# 3.3.1 But this does not, as there should be only one argument
np.array(list(range(10)), list(range(30)))


# 3.3.2 The above constitutes two separate lists. For
#       creating an ndarray, we need one list. Hence,
#       an array of two lists
np.array([list(range(10)), list(range(30))])

# 3.3.3 Same as above
np.array([range(1,4), range(5,8), range(9,12)])

# 3.3.4 OR
np.array([list(range(1,4)), list(range(5,8)), list(range(9,12))])



######################### CC. #######################################
# 4 Using list comprehensions
# Ref: http://www.secnetix.de/olli/Python/list_comprehensions.hawk
# Examples of list comprehensions (generating lists):

# 4.1 Each one of the following can also
#     be transformed to array with np.array()
[x *2 for x in [1,2,3]]
[[x*2, x/2] for x in range(5)]
[[x.upper(), x.lower()] for x in ['Abc','Cde']]



# 4.2 for and if both
[x for x in range(10) if x > 5 ]
[ x for x in range(13) if x%2 == 0]


# 4.3 for and if-else
[x if x > 5 else 4 for x in range(10)  ]


# 4.4 A little complex list comprehension transformed to array
np.array([range(i, i+3) for i in [1,4,8]])




######################### DD. #######################################

# 3.3 Broadcasting in arrays
# Ref: http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
#      https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
# Broadcasting describes how numpy treats arrays
#  with different shapes during arithmetic operations.
# General Broadcasting Rules
#  When operating on two arrays, NumPy compares their shapes element-wise.
#  It starts with the trailing dimensions, and works its way forward.
#  Two dimensions are compatible when
#    they are equal, or
#    one of them is 1
"""
Compatible: trailing dimension matches
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3

Compatible: Where dimension is 1, it is stretched to match the other:
A      (4d array):  8 x 1 x 6 x 1     => Last dimension is stretched
B      (3d array):      7 x 1 x 5     => Middle dim is stretched
Result (4d array):  8 x 7 x 6 x 5

Here are some more examples:

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5

Here are examples of shapes that do not broadcast:

A      (1d array):  3
B      (1d array):  4 # trailing dimensions do not match

A      (2d array):      2 x 1
B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched


"""


# 3.3.1
abc = np.array([1,2,3])
cde = np.array([3,4,5])
abc + cde
abc * cde

# 3.3.2
xyz= np.array([8,9])
abc + xyz

# 3.3.3 Above rule is relaxed for scalars
fx = 2
abc + fx
abc * fx

# 3.3.4 In order to broadcast, the size of the trailing axes
#       for both arrays in an operation must either be the
#       same size or one of them must be one

qx = np.random.normal(size=(2,3))
qx
qx.shape        # 2 X 3
fg = np.array([1,2,3])
fg.shape
qx + fg     # Succeeds

hg = np.array([2,3])
hg.shape
qx + hg     # Fails

mn = np.random.uniform(size = (4,5,3))
mn.shape
mn + fg     # Success
mn + fx     # Success


######################### EE. #######################################

# 3.4 Built in arrays of numpy
#      Creating arrays from sctarch
#
np.zeros(3)
np.zeros((3,5))
np.zeros((3,5), dtype=int)
np.ones((4,5), dtype=float)
np.full((4,5), 4.5)


# 3.5
# np.arange() is little similar to range()
# Both generate values within the half-open interval [start, stop)
# One generates arrays, the other list
# range() only generates integers but arange()
# can generate floats also
np.arange(3,10, 0.22)  # arange: array-range. start, stop and step
np.arange(start = 1, stop = 9, step = 2)   #  array([1, 3, 5, 7]); 9 is not included

# 3.5.1
np.linspace(50, 90, num=70)  # Generate evenly spaced numbers


# 3.6 Change array type
y = np.random.normal(4,.7,size = (5,6))
y
y.astype('int64')


# 4
# Generate Random number arrays
np.random.uniform()
np.random.uniform(size=8)
np.random.uniform(low=1.3, high=5.4, size = (10,3))

np.random.normal(size=5)
np.random.normal(loc=10, scale=5,size=10)  # scale is std dev
np.random.normal(loc=10, scale=5,size=(10,5))  # size creates a list of lists


# 4.1 Each array has attributes
#     ndim, shape (the size of each dimension),
#       and size (the total size of the array):
x=np.random.normal(loc=1,scale=2, size=(4,5,5))
x
x.ndim  # Total dimensions
x.shape # row X columns
x.size  # Total no of elements in the complete array
x.dtype # Data type


######################### FF. #######################################

# 5 Accessing arrays
#      Accessing single element
y=np.random.normal(size=(3,3))
y
y[1]          # First row
y[:2]         # First two rows
y[ :, 1]      # First column


y[3,3]        # Out of bounds
y[2,2]        # It exists
y[2][2]       # Same as above
y[-1,-1]      # Count from end. Counting begins from the last as 1 not 0
              # -0 is same as +0, both equal to 0
y[-1,1]       # Last list (row) but second element of the list (row)
y[-1,0]       # Last list but first element of the list
y[-0,1]       # First list and IInd element


x = np.random.normal(size = (6,4))
x[:4:2]      # Upto 4, in steps of 2

y = np.arange(35).reshape(5,7)
y[1:5:2,::3]  # Row wise 1 and 3 index. Column wise all values in steps of 3
              # 0, 3 , 6

# Generally speaking, what is returned when index arrays
#  are used is an array with the same shape as the index array,
#   but with the type and values of the array being indexed.
#    As an example, we can use a multidimensional index array instead:

x = np.random.uniform(size = (10,5))
x
x[np.array([[1,1,2,3]])  , : ]
x[np.array([[1,1],[2,3]]), : ]




######################### GG. #######################################


## 5.1 Slicing arrays
# Just as we can use square brackets to access individual
#  array elements, we can also use them to access subarrays
#   with the slice notation, marked by the colon (:) character.
#    The NumPy slicing syntax follows that of the standard Python
#      list; to access a slice of an array x, use this:
#           x[start:stop:step]

## 5.1.1 Create a numpy array
od=np.random.uniform(low=5,high=20,size=10)
od
od[0]    # 0th index
od[0:1]  # '1' index is not included
od[3:5]  # Start from or 3rd index upto 5th index
         # (or 5th index not included
od[:3]   # Upto (but not inclusing) 3rd index
od[2:]   # From 3nd element
od[np.array([1,3,6])]   # 1, 3rd and 6th index

# 5.1.2
g=np.arange(start=1,stop=51).reshape(5,10)  # stop= 16. 16 is not uncluded
g  # A two dimensional array
g.ndim
g.size
g[0,0]   # First elenment--top-left corner
g[0:1,0] # Still first index '1' not included
g[1:4, : ]  # Start from 1st index upto and including 3rd index
g[:4, 2:3] # Upto 3rd index and column wise start from 2nd index upto and including 2nd index
g[np.array([1,4]), np.array([2,4])] # Upto 3rd index and column wise start from 2nd index
                                    #  upto and including 3rd index


######################### HH. #######################################
"""
What is a view of a NumPy array?
As its name is saying, it is simply another way of
viewing the data of the array. Technically, that
means that the data of both objects is shared.
You can create views by selecting a slice of the
original array, or also by changing the dtype
(or a combination of both).

Only a slice is a view NOT fancy indexing
A slice has offsets, strides and counts
Fancy Indexing does not have it

"""


## 6. Sliced arrays are not copies.
#     Sliced arrays are Views
#     They work in the same memory space
#     Views share the same data but shape of a
#     view can be different from its base

fix=np.random.normal(24,size=(8,3))
fix
d = fix              # Copy by reference
d is fix             # True

d = fix [ : ]        # Full slice
d is fix             # False
d.base is fix        # False
np.may_share_memory(d, fix)      # True

d.shape = (3,8)      # d is reshaped
d.shape
fix.shape
d is fix            # False
d.base is fix       # True
np.may_share_memory(d, fix)      # True

d[0,0]              #
fix[0,0]            # Same as d[0,0]

d[0,0] = 1000
fix[0,0]

# 6.1 Fancy indexing is NOT View`
#     Fancy indexing => No offsets and strides
#     Fancy indexing is conceptually simple:
 #     it means passing an array of indices to
#        access multiple array elements at once

fix=np.random.normal(20,size=(5,4))
fix
d = fix[[2,4], [1,2]]     # Access data-points (2,1) and (4,2)
d

d is fix           # False
d.base is fix      # False
np.may_share_memory(d, fix)      # False

# 6.2 Example 2: Fancy Indexing
d = fix
h = d[d<21]
h is fix
h.base is fix
np.may_share_memory(h, fix)      # False


# 6.3 Indexing with offsets and strides are views
fix=np.random.normal(20,size=(5,4))
fix
cp=fix[0:2, :]
cp.shape
cp[0:2,1]
cp[0:2,1] = 4  # Assign every element value of 4
cp.base is fix # True
fix
np.may_share_memory(cp, fix)      # True



# 7. Make (deep) copies as:
fix=np.random.normal(24,size=(8,3))
fix
cp=fix[0:4, 0:2].copy()
cp
cp[:,:] = 4
cp
fix

# b= a  Copy by reference
a = np.random.normal(loc = 2,scale = 0.3, size=(3,4))
a
b = a
b
a[0,0] = 8
b[0,0]

b[0,0] = 5
a[0,0]


b.shape
b.shape = (4,3)
b
a.shape      # a's shape changes

# b = a[:] Shallow copy/View. Slice creates shallow copy
a = np.random.normal(loc = 2, scale = 0.3, size=(3,4))
a
c = a[ :]
c
c[0,0]
c[0,0] = 9
a[0,0]
a[0,0] = 10
c[0,0]

c.shape
c.shape = (4,3)
a.shape   #  a's shape does not change


# b[:] = a  Deep copy
a = np.random.random((5,4))
z[:] = a
z = np.empty((5,4))
z[:] = a
a[0,0]
z[0,0]
z[0,0] = 8
a[0,0]


# deep copy
a = np.random.uniform(size = (5,4))
a
b = a.copy()
b
b[0,0] = 50
a[0,0]


######################### II. #######################################

# 8. Stacking arrays vertically and horizontally
#    Concatenation, or joining of two arrays in NumPy,
#    is primarily accomplished using the routines
#     np.concatenate, np.vstack, and np.hstack.
#     np.concatenate takes a tuple or list of arrays
#      as its first argument, as we can see here:

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [99, 99, 99]
np.concatenate([x, y,z])

# 8.1 For working with arrays of mixed dimensions,
#     it can be clearer to use the np.vstack (vertical stack)
#      and np.hstack (horizontal stack) functions:

first=np.random.normal(12,size=(3,4))
first
second=np.random.uniform(12, size=(3,4))
second
h=np.hstack([first,second])
v=np.vstack([first,second])


# 9. Splitting arrays vertically and horizontally
left,right=np.hsplit(h,[1])  # Horizontal split h 1+other columns
left,right=np.hsplit(h,[2])  # Horizontal split h 2+other columns
left,right=np.hsplit(h,[3])  # Horizontal split h 3+other columns

upper,lower=np.vsplit(h,[1])  # Vertical split 1+ other rows
upper,lower=np.vsplit(h,[2])  # Vertical split 2+ other rows
upper,lower=np.vsplit(h,[3])  # Vertical split 3+ other rows


######################### JJ. #######################################


## 10. Arrays vectorized operations: Broadcasting
#      Vectorization implies parallel operations
#      at hardware level.
v1=np.random.normal(6, size=(2,3))
v2=np.random.uniform(6,size=(2,3))
v1+v2   # Element wise addition
v1*v2
v1+5
1.0/v1  # Reciprocal of element
np.power(v1,2)  # Squared v1
np.log1p(v1)    # Natural log

# 10.1
# Do array multiplication ten-times over
# with a large array. Multiply by 2
# 2 is broadcasted over the whole array
# and multiplication is done

myarr = np.arange(1000000)    # one million points
%time for _ in range(10):  myarr2 = myarr *2


# 10.1.1 Conider the following time-consuming
#        normal python operation

%time for _ in range(10):  mylist = [i * 2 for i in range(1000000)]



######################### KK. #######################################


## 12. Aggregations/summary functions
rand=np.random.random(20)
rand
np.sum(rand)
np.mean(rand)
np.min(rand)
np.max(rand)
np.std(rand)

rand=rand.reshape(5,4)
rand
np.min(rand,axis=0)   # Min along columns
np.min(rand,axis=1)   # Min along rows
rand.ravel()          # Flatten the array
rand.T                # Transpose array


# 12.1 Other summary functions
# np.var, np.median, np.percentile
a = np.array([[10, 7, 4], [3, 2, 1]])
a
# Return 10th percentile
np.percentile(a, 10, axis =0)    # It will be: 3 + (10-3)/10  OR min() + range/10
np.percentile(a, 10, axis =1)    # It will be: 4 + (10-4)/10


######################### LL. #######################################

# 13. Comparisons, Masks, and Boolean Logic
x=np.array([4,5,6,7])
x < 6    # Similary >, ==, >=, !=
x[x<6]
x=np.arange(20)
x=x.reshape(5,4)
x < 7

# How many
np.sum(x<=7)
# How mnay less than 8 along each rows
np.sum(x <= 8,axis=1)
np.any(x>8)   # Is any value greater than 8
# Are all values less than 10
np.all(x<10)
(x > 5) & (x <16)  # Boolean operations

(x > 5) | ( x > 16)  # Boolean operations

# 14. Fancy indexing is conceptually simple:
 #     it means passing an array of indices to
#        access multiple array elements at once
x =np.random.binomial(20,0.5,10)
ind=np.array([2,5,7])
x[ind]
X = np.arange(20).reshape((5, 4))
X
row=np.array([2,3])
col=np.array([0,2,3])
X[row,:]
X[:,col]
row=np.array([1,2])
X[row,col]

######################### MM. #######################################

## 15. Array sorting (row-wise)
ar=np.random.normal(size=20)
ar
np.sort(ar)
ar=np.random.normal(size=(4,5))
ar

# 16. Sort column wise
# Keep in mind that this treats each row or column
#  as an independent array, and any relationships between the
#   row or column values will be lost!
np.sort(ar,axis=0)
np.sort(ar,axis=1)



# 17. Looping over arrays
a = np.random.random((5,4))
a

for i in a:
    print(i)     # subarray is accessed

# Access each element as:

for i in a:
    for x in i:
        print(x)


# Or access each element, as:
for i in a.ravel():
    print(i)

for i in a.flat:
    print(i)

for i in a.flatten():
    print(i)




######################### NN. #######################################
# Experimentation in modules
"""
Create a file by name of 'my_module.py'
Save As: In current working folder, as: my_module.py
The file will contain following two small functions:

package vs module:
  A module is a single file that is imported under one import and used:
       import my_module
  A package is a collection of modules in directories that give a package
  hierarchy.
      from my_package.timing.danger.internets import function_of_love

"""


def sq(x):
    return (x +2) *x

def cu(x,y):
    return (x * y + 2)


"""
On another page of spyder, perform this experiment
"""

# Get your current working folder
import os
os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\python")
myfolder = os.getcwd()
myfolder
# Check that file 'my_module.py' is saved here
sorted(os.listdir())

# Add your current working folder to python
#  library search path
# https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath/471168
import sys
sys.path.insert(0, myfolder)

# Next import your module, by entering following
#  commands, one by one directly in ipython
import my_module as m
m.sq(9)
m.cu(9,8)

# Next restart ipython kernel
# Press (ctrl + .)
# And execute the following:
import sys
import os
myfolder = os.getcwd()
sorted(os.listdir())
sys.path.insert(0, myfolder)
from my_module import cu as m
m.sq(3)         # This fails
m(3)            # This fails
m(3,4)          # This succeeds

#######################################


'''
Axis in pandas

+------------+---------+--------+
|            |  A      |  B     |
+------------+---------+---------
|      0     | 0.626386| 1.52325|----axis=1----->
+------------+---------+--------+
                |         |
                | axis=0  |
                ↓         ↓

'''

## Quick decision tree program with iris
import numpy as np
from sklearn.datasets import load_iris  # data(iris) in R
from sklearn import tree                # library(C50) in R

# Start
iris = load_iris()               # Same as data("iris") in R
X = iris.data[ : , (2,3)]
X
y = iris.target
y = (y == 0).astype(np.int)
y


clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
y_pred = clf.predict(X)
np.sum(y_pred == y)/len(y)




"""
Ref: https://www.programiz.com/python-programming/package
What are python packages?
    We don't usually store all of our files
    in our computer in the same location. We
    use a well-organized hierarchy of directories
    for easier access.
    Similar files are kept in the same directory,
    for example, we may keep all the songs in the
    "music" directory. Analogous to this, Python has
    packages for directories and modules for files.
    As our application program grows larger in size
    with a lot of modules, we place similar modules
    in one package and different modules in different
    packages. This makes a project (program) easy to
    manage and conceptually clear.
    Similar, as a directory can contain sub-directories
    and files, a Python package can have sub-packages
    and modules.
    A directory must contain a file named __init__.py
    in order for Python to consider it as a package.
    This file can be left empty but we generally place
    the initialization code for that package in this file.
    Statement:
        import Game.Level.start
    means two folders: Game and Level and a file start.py
"""
