import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from NaiveBayes import CategoricalNaiveBayes

# Running NB On Data
dataset = pd.read_csv('car_data.csv', dtype = object) # Training Data

#dataset.columns
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train['target'] = y_train

X_train.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)

c = CategoricalNaiveBayes()
c.fit(X_train)
y_pred = c.predict(X_test)

c.list_of_probs

cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix\n', cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)
