import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

url='https://github.com/jbrownlee/Datasets/raw/master/pima-indians-diabetes.data.csv'

names=['preg','plas','pres','skin','test','mass','predi','age','class']

df=pd.read_csv(url, names=names)
print(df.head())

array=df.values
X=array[:,:8]
y=array[:,8]
X_train,X_test,y_train,y_test=model_selection.train_test_split(X, y, test_size=0.2, random_state=11)


model=LogisticRegression()
model.fit(X_train,y_train)

result= model.score(X_test,y_test)
print(result)


filename='diabetes_lr.pkl'
pickle.dump(model, open(filename, 'wb'))
