import joblib


model=joblib.load('diabetes_lr.pkl')

result=model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==1:
    print('diabetic')

else:
    print('non diabetic')