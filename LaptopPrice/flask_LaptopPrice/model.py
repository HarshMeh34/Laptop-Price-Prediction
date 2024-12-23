import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('i:/Machine Learning Krish/LaptopPrice/flask_LaptopPrice/output1.csv')

# print(df.head())

categorical_features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'os']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# encoder.fit(df[categorical_features])

user_input_encoded = encoder.fit_transform(df[categorical_features])
feature_names = encoder.get_feature_names_out()
user_input_encoded = pd.DataFrame(user_input_encoded, columns=feature_names)
        # Combine encoded categorical features with numerical features
df= pd.concat([user_input_encoded, df[['Ram', 'Weight', 'SSD', 'HDD', 'ppi','Price']]], axis=1)

print(df.head(4))
x= df.drop(columns=['Price'])
y= np.log(df['Price'])
x['Ram']= round(np.log(x['Ram']),2)
x['ppi']= round(np.log(x['ppi']),2 )


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

pickle.dump(lr,open('model.pkl','wb'))

