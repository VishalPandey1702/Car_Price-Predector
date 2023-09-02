import preprocessor
import pandas as pd
import streamlit as st 

car = pd.read_csv("quikr_car.csv")
car_name = preprocessor.car_name(car)
car_name = st.selectbox("Select a car name:", options=car_name)



car_company = st.text_input("Enter the company name ")
year =st.text_input("enter the year of purchase")
km_driven = st.text_input("Enter the km driven")
fuel_type = st.text_input("Enter the fuel type")

if st.button("Predict"):
    price = preprocessor.preprocess(car,car_name,car_company,year,km_driven,fuel_type)
    st.write(price)
    # print(price)