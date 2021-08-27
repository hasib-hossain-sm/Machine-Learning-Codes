# importing the Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
spam_detect_model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
spam_detect_model.fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy Check
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


#Classification report
from sklearn.metrics import classification_report
cl = classification_report(y_test, y_pred)
print(cl)
