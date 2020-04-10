# -*- coding: utf-8 -*-
"""
Last amended: 9th February, 2019
Myfolder: C:\Users\ashok\OneDrive\Documents\python\basic_lessons

# 1http://nbviewer.jupyter.org/github/WeatherGod/AnatomyOfMatplotlib/blob/master/AnatomyOfMatplotlib-Part1-Figures_Subplots_and_layouts.ipynb
# https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python
# https://matplotlib.org/users/pyplot_tutorial.html

Objectives:
	Understanding basics of matplotlib


"""

%reset -f
import numpy as np
import matplotlib.pyplot as plt

%matplotlib qt5


# 1.0 Generate some simple data
x = np.arange(start = 1, stop = 20, step = 2)    # xlim: [1,20)
y = np.arange(start = 0, stop = 10, step = 1)    # ylim: [0,20)


# 1.1 Generate more data...
x1 = np.linspace(0, 10, 100)
y1, y2, y3 = np.cos(x1), np.cos(x1 + 1), np.cos(x1 + 2)
names = ['Signal 1', 'Signal 2', 'Signal 3']



"""
The Figure is the overall window or page
that everything is drawn on. It’s the top-level
component of all the ones that you will
consider in the following points. You can
create multiple independent Figures. A Figure
can have several other things in it, such as
a suptitle, which is a centered title to the
figure. You’ll also find that you can add a
legend and color bar, for example, to your
Figure.
To the figure you add Axes. The Axes is the
area on which the data is plotted with functions
such as plot() and scatter() and that can have
ticks, labels, etc. associated with it. This
explains why Figures can contain multiple Axes.
"""

# Step 1  Create a figure:                            fig = plt.figure()
# Step 2: Add subplot (ie axes):                      ax1 = fig.add_subplot()
#         Or both 1 & 2 together                      fig, ax = plt.subplots()
# Step 3: Select plot type and draw your plot:        ax1.plot() or ax[0,1].plot()
# Step 4: Set axes properties with set_:              ax1.set_xlim(), set_title(), set_xlabel(), set_xticks()
# Step 5: Show plot:                                  plt.show()


# 1. So begin with a figure:
fig = plt.figure()

# 1.1 All plotting is done with respect to an Axes.
#     An Axes is made up of Axis objects and many other things.
# 1.2 How many axes?
ax = fig.add_subplot(111)
# 1.3 Plot
ax.plot(x,y)
# 1.4 Plot description/properties
ax.set_title("My plot")
ax.set_xlim(left = 0, right = 20)
ax.set_ylim(0,10)
ax.set_xticks(ticks = list(range(0, 20, 1)) ,minor = True)  # Specify tick points
ax.set_xlabel("X-axis")
plt.show()



# 2.You can also set in one go, as below:
#   In matplotlib.pyplot various states
#   are preserved across function calls,
#    so that it keeps track of things like
#     the current figure and plotting area,
#      and the plotting functions are directed
#        to the current axes

# 2.1
fig = plt.figure()
# 2.2
ax1 = fig.add_subplot(111)

# 2.3 Multiple plots on the SAME AXES: 'ax1'
ax1.plot(x1,y1)
ax1.plot(x1,y2)
ax1.plot(x1,y3)

# 2.4
ax1.set(title="My second plot" , xlim= [0,20], xlabel = "X-axis")
ax1.set_xticks([0, 1,2,3,4,5] )     # minor = True
plt.show()



# 3. To make further plots let us read a simple
#    data set
import os               # has OS related methods
import pandas as pd     # Pandas library

# 3.1 Set working directory
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\python\\basic_lessons")
os.chdir("/home/ashok/datasets")
os.listdir()

# 3.2 Display max number of columns
pd.options.display.max_columns = 200

# 3.3 Read and explore
fl = pd.read_csv("flavors_of_cacao.csv.zip")
fl.head()
fl.tail()
fl.dtypes
fl.shape

# 3.4 Bar chart of distribution of companies
ct = fl.company.value_counts()       # Eqt of table() in R
ct.index                             # label names
ct                                   # A sorted series of values


# 3.5 Draw barplot now
fig = plt.figure()
ax = fig.add_subplot(111)

# 3.5.1
ax.bar(ct.index[:10],             # x-values or bar-locations
       ct[:10],                   # height of bars
       color = "lightblue",       # inner-bar color (optional)
       edgecolor="darkred"        # optional
       )       # Top 10; Bottom use ct.tail()

#ax.bar?


# 3.5.2
ax.set_xticklabels(labels = ct.index[:10], rotation = 90)    # Not set_xticks()
plt.show


# 4. Draw both bar-plot and scatter plot in the same figure
#    but different axes

# 4.1 As usual create figure and also decide figsize
fig = plt.figure(figsize = (10,10))

# 4.2 Add subplot for bar-chart
ax1 = fig.add_subplot(121)
# 4.3 Add subplot for scatter plot
ax2 = fig.add_subplot(122)

# 4.4 Now plot
ax1.bar(ct.index[:10], ct[:10])
ax2.scatter(fl.rating, fl.cocoa_percent)
plt.show()


# 4.5 Let us populate multiple axes using for-loop

fig = plt.figure(figsize = (10,10))
names = ['company', 'company_location']
for i,j in enumerate(names):
	ax = fig.add_subplot(1,2,i+1)
	ct = fl[j].value_counts()
	ax.bar(ct.index[:10], ct[:10])

plt.show()


# 5. Plots within for loop
#    If a number of plots are to be made on
#    the same axis, it is more convient
#    to use plt.subplots(), as below:

# 5.1 Make bar-charts for 'company', company_location',
#     'broad_bean_origin' and 'bean_type'

fig, ax =  plt.subplots(2,2)
names = ['company', 'company_location', 'broad_bean_origin', 'bean_type']
for i,j in enumerate(names):
    ct = fl[j].value_counts()
    ax[i].bar(ct.index[:5],ct[:5])

plt.show()

# OR Use zip

fig, ax =  plt.subplots(2,2)
names = ['company', 'company_location', 'broad_bean_origin', 'bean_type']

# We have to iterate over two things: Over ax and over names
#  Use zip and ax.flat
# See topic: Multiple Axes in
#      https://github.com/matplotlib/AnatomyOfMatplotlib/blob/master/AnatomyOfMatplotlib-Part1-Figures_Subplots_and_layouts.ipynb
for i,j in zip(ax.flat,names):
    ct = fl[j].value_counts()
    i.bar(ct.index[:5],ct[:5])

plt.show()


######################## I am done ##########################################


# 1 Make your first plot
plt.plot([1,2,3,4])    # By default x is [0,1,2,3]
plt.ylabel("some numbers")
plt.show()

# 1.1 Clear rhe current figure
plt.clf()

# 1.2
plt.plot([1,2,3,4],[5,6,7,8])

# 1.3 Clear rhe current figure
plt.clf()


# 2.0 Prepare numpy array data
x = np.linspace(0, 10, 100)
# 2.1 Plot this data
plt.plot(x, x, label='linear')
# 2.2 Add a legend
plt.legend()
plt.show()

# 3. Concatenate a color string with a line style string
#  'ro'
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])  # [xmin, xmax, ymin, ymax]
plt.show()


# 3.1 Mutiple plots and formatting
plt.plot([1,2,3,4], [1,2,3,4], 'g^',
         [2,3,4,5], [2,4,5,7], 'bo') # Also use 'bs'
plt.axis([0,10, 0,10])
plt.show()


plt.gca()
plt.gcf()


"""
The following color abbreviations are supported:

==========  ========
character   color
==========  ========
'b'         blue
'g'         green
'r'         red
'c'         cyan
'm'         magenta
'y'         yellow
'k'         black
'w'         white
==========  ========

"""


# 3.3 One figure mutiple subplots
#     Create a function that decreases
#     exponentially cosine output
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

# 3.4
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)


# 4. figure(1) will be created by default,
#     just as a subplot(111) will be created
#      by default if you don’t manually specify any axes.
plt.figure(1)   # Optional will be created by default
plt.subplot(211)  # numrows, numcols, plotnum. Create 2 X 1 plots
# Note below we create two plots
#  on the same axes
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# Another plot on the second axes
plt.subplot(212)  # numrows, numcols, plotnum
                  #  IInd of the above

plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# 5. Multiple figures/window()
#     each figure can contain as many axes
#      & subplots as your heart desires:
def sinplot(t):
    return (np.sin(2.0*np.pi*t))

# 5.1 First window
plt.figure(1)

# 5.2 Data
t1= np.arange(0,180, 0.5)

# 5.3 Plot on first axes
plt.subplot(2,1,1)
plt.plot(sinplot(sinplot(t1)))

# 5.4 Plot on second axes
plt.subplot(2,1,2)
plt.plot(sinplot(t1))

# 5.5 IInd figure
plt.figure(2)
plt.plot(np.tan(sinplot(t1)))
plt.grid(True)


# 5.6 Clear rhe current figure
plt.clf()
# Clear rhe current axes
plt.cla()


# https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python


# 6. Setting legend in the figure
plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
plt.plot([1,2,3], [1,4,9], 'rs',  label='line 2')
plt.axis([0, 4, 0, 10])
plt.legend()
