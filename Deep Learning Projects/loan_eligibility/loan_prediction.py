
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (replace with your own CSV)
data = {
    "Gender":["Male","Female","Male","Male"],
    "Married":["Yes","No","Yes","No"],
    "Income":[5000,3000,4000,2500],
    "LoanAmount":[150,100,120,80],
    "Eligible":[1,0,1,0]
}
df = pd.DataFrame(data)

# Encode categoricals
for col in ["Gender","Married"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Eligible",axis=1)
y = df["Eligible"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
