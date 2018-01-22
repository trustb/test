import pandas as pd
import numpy as np
from sklearn import tree
train=pd.read_csv (r'G:/kaggle/Titanic/train.csv')
test=pd.read_csv (r'G:/kaggle/Titanic/test.csv')
train["Age"]=train["Age"].fillna(train["Age"].median())
test["Age"]=test["Age"].fillna(test["Age"].median() )
train["Sex"]=train["Sex"].apply(lambda x:1 if  x =="male"else 0 )
test["Sex"]=test["Sex"].apply(lambda x :1 if x == "male" else 0)
feature =["Age","Sex"]
dt=tree.DecisionTreeClassifier()
dt.fit(train[feature],train["Survived"])
predict_data=dt.predict(test[feature])
submission=pd.DataFrame(
    {
        "PassengerId":test["PassengerId"],
        "Survived":predict_data
    }
)
submission.to_csv('G:/kaggle/Titanic/decision_tre.csv',index=False )
