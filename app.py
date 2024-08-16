import pickle
import pandas as pd
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# loading the preprocessing models encoder and scaler

with open(r"C:\Users\HP\OneDrive\Desktop\Artificial Intellegence\Deep learning\customer churn pred\encoder.pkl","rb") as file:
    encoder=pickle.load(file)
    
with open(r"C:\Users\HP\OneDrive\Desktop\Artificial Intellegence\Deep learning\customer churn pred\scaler.pkl","rb")as file:
    scaler=pickle.load(file)
    
# loading the actual model
model=load_model(r"C:\Users\HP\OneDrive\Desktop\Artificial Intellegence\Deep learning\customer churn pred\ann.h5")




# streamlit app
# Usage Frequency	Support Calls	Payment Delay	Total Spend	Last Interaction
#Subscription Type_Basic	Subscription Type_Premium	Subscription Type_Standard	
# Contract Length_Annual	Contract Length_Monthly	Contract Length_Quarterly

st.title("Customer Churn Prediction")


age=st.slider("Select Age",18,80)
tenure=st.slider("Select Tenure",1,60)
usage_frequency=st.slider("Frequency",1,50)
support_calls=st.slider("Support Calls",0,20)
payment_delay=st.number_input("payment Delay",0,30)
total_spend=st.number_input("Total Spend",100,1000)
last_interaction=st.number_input("Last Interaction",1,30)
gender=st.selectbox("Select Gender ",["Male","Female"])
subscription_type=st.selectbox("Subscription Type" ,["Basic","Premium","Standard"])
contract_length=st.selectbox("Contract Length",["Annual","Monthly","Quarterly"])

# preprocessing the data
def preprocessing(gender,subscription_type,contract_length):
    encoded=encoder.transform([[gender,subscription_type,contract_length]])
    encoded=encoded.toarray()
    encoded_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out())
    return encoded_df

df=pd.DataFrame({
    
    
    "Age":[age],
    "Tenure":[tenure],
    "Usage Frequency":[usage_frequency],
    "Support Calls":[support_calls],
    "Payment Delay":[payment_delay],
    "Total Spend":[total_spend],
    "Last Interaction":[last_interaction]
    
     
})
processed_df=preprocessing(gender,subscription_type,contract_length)

# new df
data=pd.concat([df,processed_df],axis=1)

# scaling the data

scaled=scaler.transform(data)


#prediction

predict=st.button("Predict")

if predict:
    result=model.predict(scaled)
    if (result>0.5):
        st.write("customer is Churned")
    else:
        st.write("will not Churn")
        










