
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts=["I love this!","This is bad","Amazing product","Terrible experience"]
labels=[1,0,1,0]

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(texts)
y=labels

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=MultinomialNB()
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
