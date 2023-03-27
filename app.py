import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("/Users/eugeneleejunping/Documents/pythonProject/pythonProject_venv/data/salary_data - Sheet1.csv")
# a simple machine learning regression and fitting for the prediction feature of the streamlit app
# note that the fitting of data can cuase the subsequently predicted value to be <0; to be trained further later
# -1 is passed as a parameter so that there is no need to specify how many array is required
x:[float] = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title("Salary Predictor")
st.image("/Users/eugeneleejunping/Documents/pythonProject/pythonProject_venv/data/cat.jpeg", width=800)
nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])
if nav == "Home":

    if st.checkbox("Show Table"):
        st.table(data)

    graph = st.selectbox("What kind of graph?", ["Non-interactive", "Interactive"])

    # slider to change the data reflected
    val:int = st.slider("Fiulter data using years",0,20)
    data = data.loc[data['YearsExperience'] >= val]

    if graph == "Non-interactive":
        fig, ax = plt.subplots()
        plt.figure(figsize = (10,5))
        # plot a scatter with YearsExperience as X axis and data[salary] as Y axis
        ax.scatter(data["YearsExperience"],data["Salary"]) # these are the column titles from the csv file
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(fig)

    if graph == "Interactive":
        layout = go.Layout(
            xaxis = dict(range = [0,16]),
            yaxis = dict(range = [0,210000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'), layout=layout)
        st.plotly_chart(fig)

if nav == "Prediction":
    st.header("Know Your Salary")
    val: float = st.number_input("Enter your years of experience",0.00,20.00, step=0.25)
    val:[float] = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    # add button
    if st.button("Predict"):
        st.success(f"Your predicted annual salary is {round(pred)}")

if nav == "Contribute":
    st.header("Contribute to thee dataset. Don't worry, it is anonymous!")
    ex:float = st.number_input("Enter your years of experience",0.00,20.00)
    sal:float = st.number_input("Enter your annual salary", 0.00,10000000.00, step=1000.0)
    if st.button("submit"):
        to_add = {'YearsExperience':[ex], "Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        # mode='a' refers to append; index is not needed; header is not needed
        to_add.to_csv("/Users/eugeneleejunping/Documents/pythonProject/pythonProject_venv/data/salary_data - Sheet1.csv",mode='a',header=False,index=False)
        st.success("Submitted successfully!")
