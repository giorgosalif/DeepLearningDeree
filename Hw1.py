# import the necessary libraries
import time
import pandas as pd
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# Import the dataset
data = pd.read_csv("house-votes-84.data")

# Data preprocessing- Replace class to 1 and 2
data = data.replace('democrat',1)
data = data.replace('republican',2)

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
