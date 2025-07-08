import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


## This will predict whether the child will pass his finals or not based on the input
st.title(" Student Pass/Fail Predictor")

@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        'study_time': np.random.randint(1, 15, 100), 
        'absences': np.random.randint(0, 30, 100),
        'test_score': np.random.randint(0, 100, 100),

        
    }

    df = pd.DataFrame(data)
    #logic:
    df['pass_fail'] = ((df['test_score'] >= 37) & (df['study_time'] >= 3) & (df['absences'] < 10)).astype(int)
    return df

df = load_data()


p = df[['study_time', 'absences', 'test_score']]
r = df['pass_fail']


model = RandomForestClassifier(random_state=42)
model.fit(p,r)


st.sidebar.title(" Input Student Info")

study_time = st.sidebar.slider("Study_Time (hrs/week)", int(df['study_time'].min()), 
int(df['study_time'].max()))
absences = st.sidebar.slider("Absences", int(df['absences'].min()), 
int(df['absences'].max()))
test_score = st.sidebar.slider("Test_Score", int(df['test_score'].min()), 
 int(df['test_score'].max()))


# Predict
input_data = [[study_time, absences, test_score]]
prediction = model.predict(input_data)
result = " Pass" if prediction[0] == 1 else "Fail"

# Output
st.subheader(" Final Exam Prediction Result")
st.write(f"The student is predicted to:{result}")
