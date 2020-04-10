# -*- coding: utf-8 -*-
"""
Last amended: 15th February, 2018
My folder: C:\Users\ashok\OneDrive\Documents\python
            /home/ashok/Documents/2.datavisualization

Data file: marathon_data.csv.zip

Ref:
    1. https://seaborn.pydata.org/introduction.html
    2. https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html

    About what are splits in raunning, see:
       https://en.wikipedia.org/wiki/Negative_split

Objectives:
        1. Time and date manipulation in pandas
        2. Data manipulation using pandas
        2. Graphics in pandas using seaborn
             i.  Bivariate distribution: sns.jointplot()
             ii. Histogram: sns.distplot()
             iii.Density plot: sns.kdeplot()
             iv. Violinplot: sns.violinplot()
             v.  Box plots: sns.boxplot()
             vi. Grid of plots: sns.PairGrid()
             vii.Bar plots: sns.countplot() ; sns.barplot()
	     ix. Interpreting contour plots



What is a 'split' time in Marathon:
    Splits: A race’s total time divided
    into smaller parts (usually miles),
    is known as the splits. If a runner
    has an even split, it means they ran
    the same pace through the entire race.
    If it’s a negative split, they ran the
    second half faster than the first.
    And that’s a good thing!

    'Split hour', here, indicates time to complete
     the Ist half. Given a final time, the more
     is the 'split hour', the 'more' is negative-split.

"""

# 1.0 Reset memory and Call libraries
%reset -f

# 1.1 Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays

# 1.2 For plotting
import matplotlib.pyplot as plt      # For base plotting

# 1.3 Seaborn is a library for making statistical graphics
#     in Python. It is built on top of matplotlib and
#     numpy and pandas data structures.
#     Install latest package as:

#  conda install -c conda-forge seaborn

import seaborn as sns                # Easier plotting

# 1.4 Misc
import os

# 1.5 Show graphs in a separate window
%matplotlib qt5


######### Begin

# 2.0 Set working directory
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\python")
os.chdir("/home/ashok/datasets")
os.listdir()

# 2.1 Increase number of displayed columns
pd.options.display.max_columns = 200


# 2.2 Read data file
data = pd.read_csv("marathon_data.csv.zip")


# 2.3 Explore data
data.columns
data.dtypes                          # age is int64; This is a luxury. check np.iinfo('int64') and int8
np.iinfo('uint16')

# 2.3.1
data.describe()                      # set include = 'all' to see summary of 'object' types also
data.info()
data.shape                           # dim()
data.head()                          # head()
data.tail()                          # tail()


# 2.3.2 Levels in gender columns
data['gender'].value_counts()        # Sorted Distribution
data['gender'].unique()              # Which unique levels
data['gender'].nunique()             # No of levels



# 3. Simple time/date manipulation
#    Split datetime columns to its components
# Ref: http://pandas.pydata.org/pandas-docs/version/0.23/api.html#datetimelike-properties
# 3.1 First 'final'

data['final'] = pd.to_datetime(data['final'])      # Convert to datetime
data.dtypes


# 3.2 Now extract compoenets and create new columns
data['f_hour'] = data['final'].dt.hour.astype('uint16')           # Default int64
data['f_minute'] = data['final'].dt.minute.astype('uint16')       # default int64
data['f_second'] = data['final'].dt.second.astype('uint16')
data.dtypes
data.head()
data.shape


# 3.3 Similarly covert 'object' type to datetime type
data['split'] = pd.to_datetime(data['split'])

# 3.4.1 And extract components
data['s_hour'] = data['split'].dt.hour.astype('uint16')       # Using uni8 gives naegative automatically
data['s_minute'] = data['split'].dt.minute.astype('uint16')
data['s_second'] = data['split'].dt.second.astype('uint16')


data.head(2)
data.shape



# 4.0  Convert total time taken to seconds:
#      Calculate total duration in seconds both in 'split' and 'final' run

# 4.1
data['split_sec'] = data['s_hour'] * 3600 + data['s_minute'] * 60 + data['s_second']
data['split_sec'].head()       # Type uint16

# 4.2
data['final_sec'] = data['f_hour'] * 3600 + data['f_minute'] * 60 + data['f_second']



# 5.0 Create another column in the data, the split_fraction, which
#     measures the degree to which each runner negative-splits or
#     positive-splits the race:
#     split_frac = 0    => Both halfs equal speed
#     split_frac > 0    => IInd half took more time: +ve split
#     split_frac < 0    => Runner ran faster in IInd half: -ve split
#     Conceptually: data['split_frac'] = (data['2ndhalf'] - data["Isthalf"]) / data['total_time']

data['split_frac'] = 1 - 2 * data['split_sec']/data['final_sec']
data.head()


# 6.0 Create 3 new features: age_dec, age_cat & posOrneg
#      i)  age_dec:  You are in 20s, or 30s or 40s .....
#     ii)  age_cat:  Just cut age in three groups: young, middle, senior
#     iii) posOrneg: Label split as '+ve' or '-ve'

# 6.1 Create a new column listing whether a person
#     he in his twenties or thirtees
#     Note:  33.10//10 outputs 3, and 47//10 outputs 4. Try it yourself
#     As to what is lambda function, see:
#         https://www.w3schools.com/python/python_lambda.asp
	


# 6.2 Also cut age into three equal parts
#     ELse: pd.cut(test.days, [0,30,60], include_lowest=True)
data['age_cat'] = pd.cut( data.age,
                          bins = 3,
                          labels=["y","m","s"]
                          )


# 6.2.1
data['age_cat'].value_counts()


# 6.3 Create a new feature indicating
#     whether a split is -nev or +ve
#     First, let all values be '+ve'
data['posOrneg'] = "+ve"


# 6.3.1 Next, wherever split_frac < 0, this column be assigned '-ve' value

data.loc[data['split_frac'] < 0, 'posOrneg'] = '-ve'
data['posOrneg'].value_counts()   # Just 251 are -ve splits



##**********Example*****************
# This example illustrates how we can change values
# in some column of a dataframe based on some boolean logic

#  See: http://pandas.pydata.org/pandas-docs/stable/indexing.html#why-does-assignment-fail-when-using-chained-indexing
# 7.1 This assignment fails. Why? (see below for example)
data.loc[data['split_frac'] < 0, : ]['posOrneg'] = '-ve'

# Avoid chained assignments with fancy indexing
# Ref: https://github.com/pandas-dev/pandas/pull/5390#issuecomment-27654172
#  Fancy indexing makes a copy not a view
#  Chained assignment may make a copy of copy instead of
#   changing the values in the 'first' copy
df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 3, 6, 7]})
df2 = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3., 4., 3., 6., 7.]})
df
df2
df.loc[0]["A"] = 0
df2.loc[0]["A"] = 0
df              # This has changed
df2             # This has not changed

df.loc[1, "A"] = 0
df2.loc[1, "A"] = 0
df            # This changes
df2           # This also changes
##************ Example ends *******************



### 8. Some queries. Can be skipped
#      How many are above 80
(data.age > 80).sum()
np.sum(data.age > 80)


# 8.1 Get data for males only
data[data.gender == 'M']
data.loc[data.gender == 'M', :]


# 8.2 Get data for those above age 60
data[data['age'] > 60]
data.loc[data['age'] > 60, : ]


# 8.3 Want to see only two columns for above, say age and gender:
data.loc[data['age'] > 60, ['age', 'gender'] ]          # R-code: data[data['age'] > 60, c('age', 'gender') ]
data.loc[data['age'] > 60, data.columns.values[:2] ]



####### Plot now ############

"""
What is wrong with matplotlib?
    a. Matplotlib's API is relatively low level. Doing sophisticated statistical
       visualization is possible, but often requires a lot of code.
    b. Matplotlib predated Pandas by more than a decade, and thus is not
       designed for use with Pandas DataFrames. In order to visualize data from
       a Pandas DataFrame, you must extract each Series and often concatenate
       them together into the right format. It would be nicer to have a
       plotting library that can intelligently use the DataFrame labels in a plot.

Seaborn
    a. Good Defaults: Seaborn provides an API on top of Matplotlib that offers
       sane choices for plot style and color defaults,
    b. Simple Stat-graphs: Defines simple high-level functions for common
       statistical plot types,
    c. Pandas Dataframe: And integrates with the functionality provided by
       Pandas DataFrames.
    d. Seaborn under the hood uses matplotlib. So many matplotlib commands
       can still be used.


"""

#####################
## A. Bar charts
## Purpose: Show relative distribution of levels within a category
##          either as a count or as a mean of some continuous feature
#####################

# 9.  Plotting categorical variables: Bar plots
# 9.1 Bar plots: sns.countplot() => Bar heights are counts of levels
# Ref: http://seaborn.pydata.org/tutorial/categorical.html#categorical-tutorial

# 9.2 Distribution as Count of levels within each category
sns.countplot("age_dec", data = data)


# 9.2.1 Get more control over this graph using matplotlib functions
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111)
sns.countplot("age_dec", data = data, ax = ax)
ax.set_title("My first graph")
ax.set_xlabel("Age in decades")
plt.show()



# 9.3 Stacked Bar Charts:
#     What about count of levels of 'age_dec',
#     not in totality but gender wise
#     In seaborn we do not have stacked bar-plots
#     Note that legend appears automatically
sns.countplot("age_dec",        # Variable whose distribution is of interest
              hue= "gender",    # Distribution will be gender-wise
              data = data)



# 9.4 Plot with three categories:
#     A Stacked bar chart (two categories) is plotted on
#     multiple axes. Each axes represents one level of the
#     third category

#     catplot() introduced in version 0.9. Check
#     sns version, as:  sns.__version__
#     Upgrade seaborn as:
#     conda install -c conda-forge seaborn


sns.catplot(x="posOrneg",       # Variable whose distribution (count) is of interest
            hue="age_dec",      # Show distribution, pos or -ve split-wise
            col="gender",       # Create two-charts/facets, gender-wise
            data=data,
            kind="count"
            )


# 9.5.. barplot: sns.barplot()
#       This plot always takes two variables.
#       i)  Categorical: The category whose distribution is to be plotted
#       ii) Continuous:  The mean of this variable decides ht of categorical levels


# 9.5.1
sns.barplot(x = "age_dec",     # Data is groupedby this variable
            y= "split_sec",    # Aggregated by this variable
                               # Continuous variable. Bar-ht,
                               # by default, is 'mean' of this
            estimator=np.mean,
			hue= "gender",     # Distribution is gender-wise
            data=data
            )


"""
Error bars:
Error bars are graphical representations of the variability
of data and used on graphs to indicate the error or uncertainty
in a reported measurement. They give a general idea of how
precise a measurement is, or conversely, how far from the
reported value the true (error free) value might be. Error
bars often represent one standard deviation of uncertainty,
one standard error, or a particular confidence interval
(e.g., a 95% interval). These quantities are not the same and
so the measure selected should be stated explicitly in the graph
or supporting text.

"""



#####################
## B. Histograms and Density plots
## Purpose: Distribution of a continuous variable
#####################


"""
10. Histograms
Ref: https://seaborn.pydata.org/tutorial/distributions.html
    When dealing with a set of data, often the first thing
    one wants to do is get a sense for how the variables
    are distributed. We will give a brief introduction to
    some of the tools in seaborn for examining univariate
    and bivariate distributions.

10.3
Kernel Density Funcions:
Two types:
        i)  Single cont variable or single cont variable
            grouped by a category
        ii) Contour plots: Between two cont variables


How kde plots are drawn:
      kde plots are highly computaionally intensive. The following is
      worth reading. Briefly, at every point draw a kernel function.
      Kernel function most used is Gaussian. Then sum up ovelapping
      graphs into a smooth density function.
      See here: https://seaborn.pydata.org/tutorial/distributions.html#kernel-density-estimation
      And this example in Wikipedia:
         https://en.wikipedia.org/wiki/Kernel_density_estimation#Example

Example of kernel functions?
    https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use

"""

# 10.1. Histogram: sns.distplot()
#       Out of nearly 40,000 participants, there were
#       only 250 people who negative-split their marathon.
g = sns.distplot(data['split_frac'],
                 kde=False     # kde: Kernel density estimate plot
                 #bins = 50
                 )

# 10.2
# g: It is the Axes object with the plot for further tweaking.
#    Most seaborn plotting functions return axex object

g.axvline(0,                    # axvline and axhline are matplotlib functions
                                # Refer : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvline.html
          color="red",
          linestyle="--"
          );



# 10.3 Single variable density plot
sns.kdeplot(
           data['split_frac'],
           shade = True

           )




#####################
## C. Scatter plots
## Purpose: Relationship of two continuous variables
#####################



"""
9. Jointplots 0f two cont features or scatter plots
Re: https://seaborn.pydata.org/generated/seaborn.jointplot.html
    Draws a plot of two continuous variables.
    There are both bivariate and univariate graphs.

"""

# 11.1 Simple scatter joint plot
sns.jointplot("age",
              "final_sec",
              data,
              kind='scatter'  # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
              )




# 11.2  Why hexagonal binning?
#       Same plot as above but with hex bins
#       Read: https://www.meccanismocomplesso.org/en/hexagonal-binning/
"""
Hexagonal Binning is another way to manage the
problem of having to many points that start to
overlap. Hexagonal binning plots density, rather
than points. Points are binned into gridded hexagons
and distribution (the number of points per hexagon)
is displayed using either the color or the area of
the hexagons.
"""
sns.jointplot("age",
              "final_sec",
              data,
              kind='hex'  # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
              )



#####################
## D. Boxplots
## Purpose: How a continuous variable is conditioned by a categorical variable
#####################


# 12. Box plots: sns.boxplot()
#     Between the age-groups 20-40 there are large number
#     of outliers. This is natural as in this age-group
#     a lot of competition would exist
"""
https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
 A boxplot is a standardized way of displaying the distribution
 of data based on a five number summary:
 (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”)
 maximum whisker: Q3 + 1.5*IQR
 minimum whisker: Q1 -1.5*IQR
Suspected outliers:  between 1.5IQR and 3IQR
Confirmed outliers:  > 3IQR
"""


sns.boxplot(
            "age_cat",
            "final_sec",
             data= data
             )



#####################
## E. Heatmap
## Purpose: How value of a summary feature varies across two categorical variables
#           Plotting results of groupby when grouped on two features
#####################

# 13. Create groups by age_cat and gender
grouped = data.groupby(['age_dec', 'gender'])
# 13.1 Get summary of 'final_sec'
dt = grouped['final_sec'].agg('mean')
dt
# 13.2 Unstack
df = dt.unstack()
df
# 13.3
sns.heatmap(df)    # Plot now
sns.heatmap(df, cmap="YlGnBu")



#####################
## F. Pairplots
## Purpose: Show multiple scatter diagrams, pair-wise
#####################

# 14.0 Take a small sample from dataset
nosamples = 5000
df = data.sample(n = nosamples)

# 14.1 We will plot scatterplots of following numerical variables
#      varsforgrid = ['age', 'split_sec', 'final_sec', 'split_frac']
#      nd also include: 'gender'
#      Column indices are: 0,1,10,11,12
list(enumerate(data.columns.values))

# 14.2 Extract requisite dataset
df = df.iloc[ : , [0, 1, 10,11,12]]

# 14.3 Plot now
sns.pairplot(df)
sns.pairplot(df, hue = 'gender', kind = 'reg')

#####################
## G. Violin plots
## Purpose: Compare distributions of two cont variables
#           maybe, across levels of a category
#####################


# 15. Violinplot: sns.violinplot()
#    A nice way to compare distributions, say gender wise, is to use a violin plot
sns.violinplot("gender", "split_frac", data=data )     # x-axis has categorical variable

#15.1
sns.violinplot( "split_frac", "gender", data=data )    # y-axis has categorical variable


# 15.2. look a little deeper, and compare these violin plots as a function of age.
#       Looking at this, we can see where the distributions of men and women differ:
#       the split distributions of men in their 20s to 50s show a pronounced over-density
#       toward lower splits when compared to women of the same age (or of any age, for that matter).
sns.violinplot("age_dec", "split_frac",
               hue="gender",
               data=data,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None
               )



#####################
## H. Contour plots
## Purpose: Joint Distribution of two continuous variables
#####################


# 16 Contour plots
#      Two cont variables
data['gender'].value_counts()


# 17.1 Sunset data by gender
mdata = data[data['gender'] == 'M']
wdata = data[data['gender'] == 'W']


# 17.2 Plot two contour plots together
#      Execute both the following commnds together
sns.kdeplot(
           mdata['age'],
           mdata['final_sec'],
           cmap = 'Blues'
            )

# 17.3
sns.kdeplot(
           wdata['age'],
           wdata['final_sec'],
           cmap = 'Reds'
           )



# 18. Grid of contour plots: sns.PairGrid()
#     Draw grid of scatter plots and histograms
#     TAKES TIME TO DRAW

# 18.1 Take first a random sample of 1000 points
#      and plot grid of contour plots
#      As noted earlier, the way kde are drawn
#      requires a lot of computaion

nosamples = 5000
rs = np.random.choice(data.shape[0], nosamples)  # May replace 5000 with 1000
df = data.iloc[rs, :]
len(df)

# 18.2 Which continuous variables to be plotted?
varsforgrid = ['age', 'split_sec', 'final_sec', 'split_frac']
g = sns.PairGrid(df,
                 vars=varsforgrid,  # Variables in the grid
                 hue='gender'       # Variable as per which to map plot aspects to different colors.
                 )
# 18.3 What to plot in the diagonal? Histogram
g = g.map_diag(plt.hist)
# 18.4 What to plot off-diagonal? Kernel Density plots
g.map_offdiag(sns.kdeplot)
# 18.5
g.add_legend();



"""
Interpretaion of Contour plots:
===============================
In this contour plot, pl note:
    Contour plots will vary from student-to-student being sample
    (we will concentrate on x = age, y = final_sec)
    i)      The contours give the counts of pairs of (x,y)
    ii)     Each grid has two sets of contours--Women and Men
    iii)    Inner circle of each contour corresponds to max(count). Why?
            Match location of inner circle with the maxima of histogram
            Innercircles are therefore 'peaks'
     iv)    The width of histogram (say, for age), matches the width of contours
            both for men and women. 'Vertical width' will match corresponding
            histogram of 'final_sec' after it is rotated by 90 degrees.
      v)    In all plots above 'age', all contours have about the same width
            So also the 'vertical width' of contours across the grid.
      vi)   The innermost circle of 'men' is displayed to right-bottom from the
            inner-most circle of that of women. Meaning thereby, most women fall
            in this group of higher age and lower final_sec when compared to men.
            That is, peaks are diagonally-across
     vii)   Lastly contour-plots of Men, so to say, contain or surround,
            Women contours. It implies that 'women' form a rather narrow or closer
            group than 'men'. Closer-group implies less variance.
     viii)  You may note woman contours in other charts also.
       ix)  Suppose it were a classification problem and we were to predict 'Gender'
            Then, in some contours clear separation of peaks (or valleys) of
            Men and Women show that these data-columns can be used for distinguishing.

"""
############################### I am done ######################################
############################# Rest is upto you ##########################

# 12. Kernel Density plot: sns.kdeplot()
#   The difference between men and women here is interesting. Let's look at
#   the histogram of split fractions for these two groups:
#   The interesting thing here is that there are many more men
#   than women who are running close to an even split!

g=sns.kdeplot(                                   # 'g' is the axes object
            data.split_frac[data.gender=='M'],
            label='men',
            shade=True
            )


#12.1 How to draw the following kde plot on the same axes?
sns.kdeplot(
        data.split_frac[data.gender=='W'],
        label='women',
        shade=True,
        ax = g                # <== See this. Use the same axes object here
        )
#12.2
plt.xlabel('split_frac');








# 10.2 Other options of distribution plots
sns.distplot(data['split_frac'],
                 kde=False,
                 rug = True,         # Show vertical lines at bottom, density of points wise
                 bins = 10
                 )
# 10.2.1 Quick histogram graph from pandas plot
#        Some graphs can be very revealing
dm = data.loc[data['gender'] == "M", ['age']].reset_index(drop = True)
dw = data.loc[data['gender'] == "W", ['age']].reset_index(drop=True)
df = pd.concat([dm,dw], axis = 1)
df.columns = ['mage', "wage"]
df.head(2)
df.plot(kind = 'hist', subplots = True)



# Method I
sns.distplot(data['split_frac'],
                 kde=True,
                 rug = True,
                 bins = 10
                 )


# 9.3 Drawing multiple plots on the same axes

# 9.3.1   First draw a jointplot is a plot between split_sec and final_sec
g = sns.jointplot("split_sec", "final_sec", data, kind='hex')

# 9.3.2   Next use g axes object:
#         On joint-axis, plot another graph
#         The dotted line shows where someone's time would lie if they ran
#         the marathon at a perfectly steady pace. The fact that the
#         distribution lies above this indicates (as you might expect)
#         that most people slow down over the course of the marathon.
#         ie final > 2 * split

g.ax_joint.plot(                # Plot y versus x as lines and/or markers.
               np.linspace(4000, 16000),    # x-axis
               np.linspace(8000, 32000)     # y-axis
               )                            # For even split every point on
                                            # the line is (x, 2*x) or (x,y)


# 7.4
# First define descending_order
# value_counts() are generally sorted
descending_order = data['age_dec'].value_counts().index

sns.countplot("age_dec",        # Variable whose distribution is of interest
              hue= "gender",    # Subset: Distribution will be gender-wise
              data = data,
              order = descending_order
              )


# 3.5 Calculate time difference between 'split' and 'final'
#     Reword data['diff'] as data[IIndhalf]

data['2ndhalf'] = data['final'] - data['split']
data.dtypes           # Note the datatype of data['2ndhalf']
                      # It is timedelta64

# 3.6 Some timedelta operations: airthmatic operations
# 3.6.1
data['2ndhalf'][0] - data['2ndhalf'][1]
# 3.6.2
data['2ndhalf'][0].total_seconds()
# 3.6.3
data['final'][0].total_seconds()    # Not permitted on datetime
# 3.6.4
data['2ndhalf'] * 6           # Six times
data.head()
