# import the necessary libraries
import time
import numpy as np
import pandas as pd
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# Import the dataset
data = pd.read_csv("house-votes-84.data")

# Data preprocessing- Replace class to 1 and 2
data = data.replace('democrat',1)
data = data.replace("republican",2)

# Train test split
train, test = train_test_split(data,test_size=.33,random_state=42)

# Download the classifier
clf = lw.RIPPER(random_state=42)

# Training time
training_start_ruleset = time.time()

# Train our model based on the train dataset
clf.fit(train,class_feat='ClassName',pos_class=1)
training_finished_ruleset = time.time()
elapsed_time_rulerset = training_finished_ruleset - training_start_ruleset

# X_test , y_test
X_test = test.drop('ClassName',axis=1)
y_test = test['ClassName']

# Metrics
precision = clf.score(X_test,y_test,precision_score)
accuracy = clf.score(X_test,y_test,accuracy_score)
recall = clf.score(X_test,y_test,recall_score)

# F1 score
cond_count = clf.ruleset_.count_conds()
# Prints statements
print('precission' , precision, ' accuracy ', accuracy , 'recall',recall)
print( f'time elapsed {elapsed_time_rulerset}')
print(clf.ruleset_)


# -Up until here was the code provided
#2a.) By carefully looking at the ruleset the model has learned, and given that the “?” value SHOULD represent a “Don’t Know” value on how a congressman/woman voted,
# what can you say about how the package handles “?” values?

#Answer: The model should  learn patterns based on the available information and handle "?" values appropriately during rule generation. It is crucial to keep those ? values to have better results.
# RIPPER may generate rules that involve "?" values when it finds that including such rules improves the accuracy of the model. For example, it might learn rules like
# "If feature X is '?' and feature Y is 'n', then predict class 1 (Democrat)."


# Replacing "?" with the mean might disrupt this process and lead to less accurate rules.


#2b. Use pandas to replace the “Don’t Know” values with np.nan (Not-A-Number in the numpy library). Re-run the fit method of the Wittgenstein package RIPPER classifier. What accuracy do you obtain now?

#Replace ? with nan in the original dataset
data = data.replace('?',np.nan)

X = data.drop('ClassName',axis=1)
y =data['ClassName']

# Do the train-test split with the same random_state to split my dataset in the same way, to be able to compare the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training time
training_start_ruleset = time.time()

# Train the model with np nan on it
clf.fit(X_train,y_train,pos_class=1)
training_finished_ruleset = time.time()
elapsed_time_rulerset = training_finished_ruleset - training_start_ruleset

# Metrics
precision = clf.score(X_test,y_test,precision_score)
accuracy = clf.score(X_test,y_test,accuracy_score)
recall = clf.score(X_test,y_test,recall_score)
cond_count = clf.ruleset_.count_conds()
# Prints statements
print('precission ' , precision, ' accuracy  ', accuracy , 'recall ' ,recall)
print( f'time elapsed {elapsed_time_rulerset}')
print(clf.ruleset_)
# I am having worse results using np.nan versus having ? in my dataset

# Question 3 .Replace Nan values with the mean value of the column. Fit the model and check the results

for column in data.columns:
    mode_value = data[column].mode().iloc[0]
    data[column].fillna(mode_value, inplace=True)

X = data.drop('ClassName',axis=1)
y =data['ClassName']

# Do the train-test split with the same random_state to split my dataset in the same way, to be able to compare the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training time
training_start_ruleset = time.time()

# Train the model with np nan on it
clf.fit(X_train,y_train,pos_class=1)
training_finished_ruleset = time.time()
elapsed_time_rulerset = training_finished_ruleset - training_start_ruleset

# Metrics
precision = clf.score(X_test,y_test,precision_score)
accuracy = clf.score(X_test,y_test,accuracy_score)
recall = clf.score(X_test,y_test,recall_score)
cond_count = clf.ruleset_.count_conds()
# Prints statements
print('precission ' , precision, ' accuracy  ', accuracy , 'recall ' ,recall)
print( f'time elapsed {elapsed_time_rulerset}')
print(clf.ruleset_)
