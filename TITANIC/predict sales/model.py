import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('train.csv')

dataset=df = dataset.drop(columns=['PassengerId', 'Name','Ticket','Embarked'])
dataset['Cabin'].fillna("None", inplace = True)
dataset['Age'].fillna(dataset.groupby('Pclass')['Age'].transform('median'), inplace=True)
x=dataset.loc[:, dataset.columns != "Survived"]
x= pd.get_dummies(x)

y= dataset["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(max_depth=10, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=50, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

pickle.dump(rf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))



'''
dataset['rate'].fillna(0, inplace=True)

dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500]]))
'''