# Last amended: 15th Februray, 2019
# Myfolder: D:\data\OneDrive\Documents\decision_trees
# Ref: https://towardsdatascience.com/a-guide-to-decision-trees-for-machine-learning-and-data-science-fe2607241956
# Objectives:
#            i) To quickly create a decision tree
#           ii) To see the decision tree

# 1.0 Call libraries
%reset -f
import numpy as np
import pandas as pd
import os

# 1.1 Call sklearn libraries
# 1.1.1 Convert target values from string to integers
from sklearn.preprocessing import LabelEncoder as le
# 1.1.2 Split data into train and test data
from sklearn.model_selection import train_test_split
# 1.1.3 Import class DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as dt

# 1.2 For tree visualization
"""
Ref: https://stackoverflow.com/questions/33433274/anaconda-graphviz-cant-import-after-installation
Install on Anaconda using following two commands, as:

  conda install -c anaconda graphviz , python-graphviz
  conda install python-graphviz

"""
#import graphviz



# 2.0 Load in our dataset
#path = "C:\\Users\\ashok\\Desktop\\chmod\\lubuntu_machinelearning_I\\today"
path = "/home/ashok/datasets/iris"
os.chdir(path)
iris = pd.read_csv(
                  "iris_wheader.csv",     # Data is without headers
                   header = None,
                   names = ["c1","c2","c3", "c4", "target"]
                   )


iris.head()

# 2.1 Separate predictors and target
X = iris.iloc[: , 0:4]    # Predictors: First 4 columns
y = iris.iloc[:, 4]       # Target: Last, 5th column


# 2.2 Split X and y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3)
X_train.shape
X_test.shape
y_train[:4]


# 2.2 Encode y_train from object to inetger
enc = le()                         # Create an instance of class labelencoder
enc.fit(y_train)                   # Let the object learn data
y_tr = enc.transform(y_train)      # Let it encode
y_tr


# 3. Start modeling
# 3.1 Initialize our decision tree object.
#     Supply relevant parameters
ct = dt( criterion="gini",    # Alternative 'entropy'
         splitter="best",     # Alternative 'random'
         max_depth=None       # Alternative, specify an integer
                              # 'None' means full tree till single leaf
         )


# 3.2 Train our decision tree
c_tree = ct.fit(X_train,y_tr)

# 4.0 Make predictions of test data
# 4.1 First transform y_test into inetgers
#     just as in y_tr
#     We use the already trained enc() object
y_te = enc.transform(y_test)

# 4.2 Now make prediction
out = ct.predict(X_test)
out

# 4.3 Get accuracy
np.sum((out == y_te))/out.size


# 5.0 Which features are important
fi = ct.feature_importances_
fi

# 5.1 Get a list
list(zip(X.columns, fi))

######### Drop 'c2' and repeat above steps #############

# 6. Start modeling
# 6.1 Initialize our decision tree object
ct1 = dt( criterion="gini",    # Alternative 'entropy'
         splitter="best",     # Alternative 'random'
         max_depth=None       # Alternative, specify an integer
                              # 'None' means full tree till single leaf
         )


# 6.2 Train our decision tree (tree induction + pruning)
ct1.fit(X_train[['c1', 'c3', 'c4']],y_tr)


# 6.3 Now make prediction
out = ct1.predict(X_test[['c1', 'c3', 'c4']])
out

# 6.4 Get accuracy
np.sum((out == y_te))/out.size

# 7.0 Which features are important
fi = ct1.feature_importances_
fi
list(zip(X[['c1', 'c3', 'c4']].columns, fi))

########################## I am done ############################

from sklearn.tree import export_graphviz
#3
feature_names=[ 'c1','c2','c3','c4']
class_names = ['target']
dot_data = export_graphviz(c_tree, out_file=None,
                     feature_names=feature_names,
                     class_names=class_names,
                     filled=True, rounded=True,
                     special_characters=True)

# 3.1
graph = graphviz.Source(dot_data)

# 3.2 A pdf file is created in your current folder
graph.render("iris")

"""
Graph explanation:
Top level
=========
		petal width <= 0.8
		gini = 0.667		=>As 1/3rd smaples exist of each type, GI=0.667
		samples = 150
		value =[50,50,50]
		class=setosa		=> Branch on this class(True/False)

Next level(two branches)
=========================

	gini=0		=>Being pure, no impurity here. GI=0
	samples=50
	value=[50,0,0]
	class=setosa
				petal width<=1.75
				gini = 0.5	=>Being 50:50, impurity is max, 0.5
				samples = 100
				value=[0,50,50]
				class=versicolor

Overall impurity = (50/150) * 0 + (100/150) * 0.5 = 0.33

#########################
