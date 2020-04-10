'''
Last amended: 20/01/2019
Myfolder: C:\Users\ashok\OneDrive\Documents\python
	  /home/ashok/Documents/1.basic_lessons

Examples of python usage:
    	(https://www.python.org/about/apps/)
    	i) Web and Internet Development
    	ii) Scientific, Engineering and Numeric
    	iii) Education (for teaching programming)
    	iv) Desktop GUIs (widgets, gtk+)
    	v)  Software Development (database related applications)
    	vi) Business Applications (ERP/e-commerce systems)


Objectives:
	i)   Learn python data structures
	ii)  Opening and reading file
	iii) Iterating over sequences with for-loop
	iv)  List/dictionary comprehension
	v)   Writing functions in python
	vi)  Writing classes in python



References:
	https://docs.python.org/2/tutorial/introduction.html#unicode-strings
       	http://www.datacarpentry.org/python-ecology-lesson/
       	http://www.datacarpentry.org/python-ecology-lesson/00-short-introduction-to-Python
       	http://www.datacarpentry.org/python-ecology-lesson/00-short-introduction-to-Python
'''

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

# 1.0
     # ipython magic command to clear memory


# 2. OS related operations
import os
os.getcwd()
#os.chdir("c:\\users\\ashok\\Documents")
#os.chdir("/home/ashok/Documents")
os.listdir()  # Directory contents


# 2.1 Joining path and filenames correctly
path = "/home/ashok/Documents"
fname = "readme.txt"
os.path.join("/home/ashok/Documents", fname)


# 2.2 How to read a file
file = os.path.join("/home/ashok/Documents", fname)
s = open(file)
data = s.read()
data
s.close()      # File must be closed to release resources back to system


# 2.3 Look at the file contents
data


# 3.1. Variable types in python need not be
#     declared beforehand just as in R
#     python is dynamically-typed language

x=5                   # Type of x is integer
x="abc"               # type of x changed but no complaints

# A Python Integer Is More Than Just an Integer
#  A Python integer is a pointer to a position in
#    memory containing all the Python object information,
#     including the bytes that contain the integer value.
#  https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html
x = 10
x.<tab>               # This shows that 'x' is not merely
                      #  a raw-integer but stores much more information


type(x)               # R command is class()
type(y)
type(data)


# 3.2 Operators
#  +, -, /, *, %

"abc" + "cde"    # Concatenation
2 + "abc"        # Error
40 % 5           # Modulo: Gives remainder
2 ** 5           # 32; Power operator

# 3.3 Comparison operators
#     <, >, ==, !=, <=, >=

5 > 3
5 == 5
5 == "abc"
"abc" == "abc"
"abc" > "cde"
"abc" >= "abc"

# 3.4 Logic Operators
#     and , or, not

not(True)              # False
True and False         # False
False or not(False)    # True


########### Sequence Types #####################
##### 4. Lists, Tuples & dictioaries ############
#  sequence is the generic term for an ordered set.
# Some sequence classes in python are: lists, tuples, strings
#   and range objects

 '''
Lists
=====
Python knows a number of compound data types, used to group together
other values. The most versatile is the list, which can be written
as a list of comma-separated values (items) between square brackets.
Lists might contain items of different types, but usually the items
all have the same type.

https://docs.python.org/3/tutorial/datastructures.html

'''


# 4.1 Lists
numbers=[1,2,5]
numbers[2]               # Indexing starts from 0
numbers[2] = 89          # List value is reassigned


# 4.2 Useless list
ul = [ 1, "abc", 23.34, 2, "abc"]
type(ul)



# 4.3 List of methods that apply to list
dir(numbers)	           # numbers.__sizeof__()


# 4.3 Append a number or list to list
#     Append acts to create 'stacks' of lists
ex=[2,9]
numbers.append(ex)
numbers.append(10)
numbers

# 4.3.1 Note the difference between
#         append() and extend():

ex = [2,9]
numbers = [6,7,8]
numbers.append(ex)
numbers


# 4.3.2
numbers= [6,7,8]
ex = [2,9]
numbers = [6,7,8]
numbers.extend(ex)
numbers


# 4.4 Popping out from list. Last in first out
numbers.pop()
numbers


# 4.5 Deleting any list item in between
gef = list(range(3,21,2))
gef


# 4.6 What is the poisition of value 13
gef.index(13)
gef.pop(5)
gef

# 4.7 Iterate over contents of list
#     For iterator vs iterable
#     Ref: https://stackoverflow.com/questions/9884132/what-exactly-are-iterator-iterable-and-iteration

for num in numbers:             # s =numbers__iter__() ; list(s)
	print(num)


# 4.7 Or as here
result = 0
for i in range(100):           # range(5) is an iterable just as list is
    result += i

result


# 4.8 Single line for loop
#      Enclosed in square brackets
#       chr(97) is 'a'
#     List comprehension

squares = []
for x in range(10, 20):
    squares.append(x**2)

squares

# 4.81  List comprehension
squares = [x * x for x in range(10, 20)]
squares


# 4.82
squares = [(x,x * x) for x in range(10, 20)]
squares


# 4.83
x=[chr(i) for i in range(97,100) ]
x


# 4.9  Heterogeneous lists
#        But this flexibility comes at a cost: to allow
#         these flexible types, each item in the list must
#          contain its own type info, reference count, and other
#            information–that is, each item is a complete Python object.
 #            It can be much more efficient to store data in a fixed-type array.
 #             For example np.array()
 # https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html

L3=[True, "abc", 1, 2.9]
type(L3)
for item in L3:
    print(type(item))


# 4.10 Delete a list
n = [1,2,3]
del n[:]
n              # n is an empty list
del n          # n is deleted
n




'''
Tuples
======
A tuple is similar to a list in that it's an ordered
sequence of elements. However, tuples can not be
changed once created (they are "immutable").
Tuples are created by placing comma-separated
values inside parentheses ().
'''


# 5
t=(3,4,6,8)
t[1]
t[1] = 6	# Gives error


# 5.1
a = 1,7     # This is also a tuple
type(a)



# 5.2 zipping tuples

t1 = (1,4,5,6)
t2=("a","b","c","d")
zip(t1,t2)
list(zip(t1,t2))
tuple(zip(t1,t2))


for idx, x in zip(t1,t2):
    print(idx)
    print(x)

# 5.3 enumerate returns position of an element and also element
for idx, x in enumerate(zip(t1,t2)):
    print(idx)
    print(x)



 '''
A dictionary is a container that holds unordered pairs
of objects - keys and values.
'''

# 6
d = { 'a' : 9, 'b' : 6}
d['a']
d['a'] = 999
d['b']
d['c'] = 9	# Add a key to d


# 6.1
len(d)			# Function on dict
type(d)
d.keys()		# Methods
d.values()


# 6.2
d.items()
[(i,j) for i , j in d.items()]


del d['a']		# Delete ke:value pair of 'a'


"""
Range
=====
Generate a sequence of numbers as a list
range(start,stop,step)
  step can be negative
"""

## 7. Range types: Occupies much less memory than lists
 #     range type represents an immutable sequence of numbers
import sys

a = range(10)
b =range(10,15)
c = range(10,100000)
import sys
sys.getsizeof(a)      # Check memory occupied
sys.getsizeof(c)
sys.getsizeof(list(c))


# 7.1
5 in a
20 in a
len(a)

# 7.2 Slicing sequences

d = range(200)
d[9:103:8]            # Output starts from 9th index upto 103 in steps of 8


# 7.3
d = "This is a good sentence. I like it."
d[2:15:3]            # Start from 2nd index

# 7.4
x = list((3,8,10,2,34,100,67,23))
x[1:4:2]         # Note only one number is output
               # The slice from i to j is: i <= k < j.

x[-3]          # Index from end start from 1
               # In R it means except 3rd
               # Here it means 3rd from end



"""
strings
=======
String module provides a number of methods to manipulate
strings.
https://www.tutorialspoint.com/python/python_strings.htm
https://docs.python.org/3/library/stdtypes.html#typesseq
"""

## 8. String sequence type is immutable
s = 'xyabcdefxy'
x = 'abcdef'

# 8.1 Operations on strings
'x' in s
'xy' in s
'xa' in s
'fx' in s
'xya' in s
'xyz' not in s
s + x
s * 2
s +=x
s

# 8.2 Methods on strings
#      upper, lower, split, replace, isdigit, isalnum
#      capitalize

s.isalnum()
s.capitalize()

s.split("b")  # Split separator is 'b'
s.replace('xy', 'zz' )
s.replace('xy', 'zz', 1 )  # At most one place
s.upper()
s.upper().lower()
s.isdigit()

h = '123'
h.isdigit()
s.isalnum()
s.capitalize()


# 8.3 Accessing strings
s
s[0]
s[1:3]
s[1:6:2]   # access index 1, 3 and 5
len(s)
s


# 8.4 Strings are immutable
s[0] = '9'    # Returns error


# 8.4 for loop
for i in s:
    print(i)




"""
sets
====
https://docs.python.org/2/library/sets.html
# Sets module provides classes for manipulation of
#  unordered sequence of unique elements
# Common uses include: removing duplicates,
# and set operations such as:
#    intersection, union, difference, and symmetric difference.
"""

# 9.1
s = {"Delhi", "Delhi", "Kanpur", "Kanpur", "ooty", "ooty"}
s
type(s)
len(s)


# 9.2
t = {"Delhi", "Jaipur", "Ahmedabad", "Lucknow", "Chandigarh", "Hissar"}
len(t)
s.union(t)                # All elements
s.intersection(t)         # Common element


# 9.3
s.difference(t)           # Remaining back in s
t.difference(s)           # Remaining back in t


# 9.4
"Delhi" in s              # membership test
                          #  Also:   s is t


# 9.5 Subset
{"Delhi", "Jaipur"}.issubset(s)
{"Delhi", "Kanpur"}.issubset(t)


# 9.6 Superset operation
s.issuperset({"Delhi", "Kanpur"})



"""
Functions
=========

Functions vs Methods
A method refers to a function which is part of a class.
You access it with an instance or object of the class.
A function doesn’t have this restriction: it just refers
to a standalone function. This means that all methods
are functions but not all functions are methods.

"""

# 10 Defining function

## 10.1 A python function
def squareof(x):
    return x * x

# 10.2 Multi return function
#      % is modulus
def squareof(x,y):
    return x * x, x%y

squareof(2,4)
squareof(y = 4,x = 2)       # Call with keyword arguments in any order



## 10.3 An R function
g = function(x) {
        return (x * x)
        }
`
# 10.4 Square contents of a list
hj=[]
 def sq(xx):
    for i in xx:
        re=i * i
        hj.append(re)

sq(numbers)
hj


# 10.5 Function returns three values
#     Input to function can be list or Series
def myfunction (c,d):
	l=np.mean(c)
	g=np.median(d)
	h=[l,g]
	return h,l,g


c=[1,2,3]
d=[4,5,6]
x,y,x= myfunction(c,d)
x
y
z



## 10.6 A python function
def squareof(x):
    return x * x

def squareof(x,y):
    return x * x, x%y

## An R function
g = function(x) {
        return (x * x)
        }



# 10.7 Variable number of arguments
def var(*myargs):
    for i in myargs:
        print(i)
    return (np.sum(myargs))

var(2,7,8)

# 10.8 Anonymous functions are also called
#      lambda functions in Python because
#      instead of declaring them with the
#      standard def keyword, you use the lambda keyword.

myfun = lambda x: x*8
myfun(9)


########################
# 11. Some common operations on sequence types

x in s             # True if an item of s is equal to x, else False 	(1)
x not in s 	     # False if an item of s is equal to x, else True 	(1)
s + t 	            # the concatenation of s and t 	(6)(7)
s * n or n * s 	  # equivalent to adding s to itself n times 	(2)(7)
s[i] 	            # ith item of s, origin 0 	(3)
s[i:j] 	         # slice of s from i to j 	(3)(4)
s[i:j:k] 	         # slice of s from i to j with step k 	(3)(5)
len(s) 	         # length of s
min(s) 	         # smallest item of s
max(s) 	         # largest item of s
s.index(x[, i[, j]]) 	# index of the first occurrence of x in s (at or after index i and before index j) 	(8)
s.count(x) 	          # total number of occurrences of x in s



#12 Classes in python

class abc:
    a = 3
    b= 5
    def omg(self):
        return(self.a * self.b)


k = abc()
k.a = 5
k.b = k.a
k.omg()


class yo1:
    value = 5
    def __init__(self, value):
        self.value = value

    def fog(self, nv):
        self.value = nv


ok = yo1(200)
ok.value

ok.fog(300)
ok.value



########## FINISH ################
"""
About python's usage:
    Ref: https://www.python.org/about/

    1. Web and Internet Development

        Frameworks such as Django and Pyramid.
        Micro-frameworks such as Flask and Bottle.
        Advanced content management systems such as Plone and django CMS.

    2. Database Access
    3. Desktop GUIs
    4. Scientific & Numeric

        SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of
        open-source software for mathematics, science, and engineering.
        In particular, its core packages are:
            numpy: An N-dimensional array package provides sophisticated functions
            pandas: high-performance, easy-to-use data structures and data analysis tools
            ipython:
            matplotlib

        scikit-learn Machine Learning in Python
           http://scikit-learn.org/stable/index.html

    5. Education
    6. Network Programming
    7. Software & Game Development
"""
