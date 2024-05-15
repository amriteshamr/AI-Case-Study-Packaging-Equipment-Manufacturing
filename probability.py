import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

shape = ['circle', 'triangle', 'square']
crust_size = ['thick', 'thin']
crust_shade = ['grey', 'white', 'dark']
filling_size = ['thick', 'thin']
filling_shade = ['grey', 'white', 'dark']
result = [1,0]

num_rows = 20


data = {
    'shape': np.random.choice(shape, num_rows),
    'crust_size': np.random.choice(crust_size, num_rows),
    'crust_shade': np.random.choice(crust_shade, num_rows),
    'filling_size': np.random.choice(filling_size, num_rows),
    'filling_shade': np.random.choice(filling_shade, num_rows),
    'result': np.random.choice(result, num_rows)
}


df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df)


X = df_encoded.drop('result', axis=1)
y = df_encoded['result']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

