import pickle


model=pickle.load(open('diabetes_lr.pkl','rb'))

result=model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==1:
    print('diabetic')

else:
    print('non diabetic')