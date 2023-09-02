import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import  make_column_transformer
def preprocess(car,car_name,car_company,year,km_driven,fuel_type):
# car = pd.read_csv("quikr_car.csv")
    car = car.dropna()
    car.isnull()
    car['name'].unique()
    backup = car.copy() # creating the copy of the data set
    car=car[car['year'].str.isnumeric()] # here we take those column which is numeric
    car['year']= car['year'].astype(int)
    # car= car[car['Price'].str.isnumeric()]
    # OR
    car = car[car['Price'] != "Ask For Price"]
    car['Price']=car['Price'].str.replace(',','') # it will remove the comma from the values
    car['Price'] = car['Price'].astype(int) # it will convert  it into the integer
    car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','') # it will split ,remove km ,replace ","
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven']=car['kms_driven'].astype(int)
    car['name'] = car['name'].str.split(' ').str.slice(0,3,1).str.join(' ')
    car.reset_index()
    car.describe()
    car[car['Price']>6e6].reset_index(drop=True)


    car_name_from_list = pd.DataFrame(car['name'])


    car.reset_index()
    X = car.drop(columns="Price")
    Y = car['Price']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    ohe = OneHotEncoder()
    ohe.fit(X[['name','company','fuel_type']])
    column_trains = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
    lr = LinearRegression()
    pipe = make_pipeline(column_trains,lr)
    pipe.fit(X_train,Y_train)
    y_pred = pipe.predict(X_test)
    r2_score(Y_test,y_pred)

    score = []
    for i in range (1000):
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=i)
        lr = LinearRegression()
        pipe = make_pipeline(column_trains,lr)
        pipe.fit(X_train,Y_train)
        y_pred = pipe.predict(X_test)
        score.append(r2_score(Y_test,y_pred))

    # np.argmax(score)
    # score[661]
    price = pipe.predict(pd.DataFrame([[car_name,car_company,year,km_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    # price = pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2009,10000,'Diesel']],columns=['name','company','year','kms_driven','fuel_type']))
    return price

def car_name(car):
    car = car.dropna()
    car.isnull()
    car['name'].unique()
    backup = car.copy() # creating the copy of the data set
    car=car[car['year'].str.isnumeric()] # here we take those column which is numeric
    car['year']= car['year'].astype(int)
   
   
    car = car[car['Price'] != "Ask For Price"]
    car['Price']=car['Price'].str.replace(',','') # it will remove the comma from the values
    car['Price'] = car['Price'].astype(int) # it will convert  it into the integer
    car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','') # it will split ,remove km ,replace ","
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven']=car['kms_driven'].astype(int)
    car['name'] = car['name'].str.split(' ').str.slice(0,3,1).str.join(' ')
    car.reset_index()
    car.describe()
    car[car['Price']>6e6].reset_index(drop=True)


    car_name_from_list = pd.DataFrame(car['name'])
    return car_name_from_list


