<span style="color:#1E90FF; font-size:50px; font-weight:bold;">Anti-Spam Analytics</span>
# <span style="color:black; font-size:40px; font-weight:bold;">Email</span> <span style="color:#FF0000; font-size:40px; font-weight:bold;">Spam</span> <span style="color:#32CD32; font-size:40px; font-weight:bold;">Ham</span> <span style="color:black; font-size:35px; font-weight:bold;">Detection</span>

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.display import display,HTML

data = pd.read_csv(r"C:\Users\FLASH\Desktop\FLASH\Business Analytics\Machine Learning\Logistic Regression\Email Spam Ham Detection\mail_data.csv")
print(data)

<span style="color:#FF1493; font-size:40px; font-weight:bold;">Bag of words</span> <span style="color:#00000; font-size:20px; font-weight:bold">(Converting Emails/Text/Strings to Numeric Vectors)</span>

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['Message'])
print(x.toarray())

<span style="color:#32CD32; font-size:35px; font-weight:bold;">Mapping</span> 
<span style="color:#000000; font-size:20px; font-weight:bold;">(Converting Categorical data (Ham|Spam) into Numerical data (0|1))</span>

y = data['Category'].map({'ham':0, 'spam':1})

# Rows & Cols
x.shape, y.shape

<span style="color:#000000; font-size:30px; font-weight:bold;">Splitting data</span>

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

<span style="color:#E91E63; font-size:30px; font-weight:bold;">Logistic Regression</span>

model = LogisticRegression()
model.fit(x_train, y_train)

<span style="color:#9B59B6; font-size:30px; font-weight:bold;"> Predicted Outputs </span>


y_pred = model.predict(x_test)
print("Predicted Spam & Ham Emails", y_pred[-15:])
print("Actual Spam & Ham Emails", y_test[-15:])

<span style="color:#2ECC71; font-size:30px; font-weight:bold;"> Model Performance Evaluation </span>
# <span style="color:#1E90FF; font-size:30px; font-weight:bold;"> Accuracy_Score: 98.4% </span>

accuracy = accuracy_score(y_pred, y_test)
print(accuracy)

print(len(vectorizer.vocabulary_))

<span style="color:#00BCD4; font-size:30px; font-weight:bold;"> Predicting Random Emails </span>


new_email = '''Limited time offer: get a free smartphone today, just reply with your contact information.'''
z = vectorizer.transform([new_email])
new_pred = model.predict(z)

if new_pred[0] == 1:
    display(HTML('<span style="color:#FF0000; font-size:30px; font-weight:bold;">🚨 Spam Detected 🚨(Phishing Alert)</span>'))
elif new_pred[0] == 0:
    display(HTML('<span style="color:#008000; font-size:30px; font-weight:bold;">✅ Legitimate/Safe Email</span>'))
else:
    display(HTML('<span style="color:#000000; font-size:30px; font-weight:bold;">❌ Email Unparseable</span>'))
