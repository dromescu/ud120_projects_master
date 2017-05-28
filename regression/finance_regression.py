#!/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""


import sys
import pickle
"""
RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
  warnings.warn(mesg, RuntimeWarning)
"""
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.model_selection import train_test_split



sys.path.append("../tools/")
#sort_keys = '../tools/python2_lesson06_keys.pkl'
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )

### list the features you want to look at--first item in the
### list will be the "target" feature
#features_list = ["bonus", "salary"]
features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(feature_train,target_train)
print("Slope:", reg.coef_)
print("Intercept:", reg.intercept_)
print("Score Training:", reg.score(feature_train, target_train))
print( "Score Testing:", reg.score(feature_test, target_test))


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color )
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color )

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

reg2 = LinearRegression()
reg2.fit(feature_test, target_test)
print("Test Samples: ")
print("Slope:", reg2.coef_)
print("Intercept:", reg2.intercept_)
print("Score Training:", reg2.score(feature_train, target_train))
print( "Score Testing:", reg2.score(feature_test, target_test))

plt.plot(feature_train, reg2.predict(feature_train), color="r")


plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
