import streamlit as st
import pandas as pd
import pickle

# 加載模型和pipeline
model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

def predict(entry):
    # 使用pipeline轉換數據
    entry_transformed = pipeline.transform(entry)

    # 進行預測
    prediction = model.predict(entry_transformed)

    # 構建預測結果
    res = {'predictions': {}}
    for i in range(len(prediction)):
        res['predictions'][i+1] = int(prediction[i])

    return res

# 添加標題
st.title("Machine Learning Model Prediction")

# 使用Streamlit的輸入小部件來收集單獨的輸入數據
passenger_id = st.number_input("PassengerId", value=1987)
pclass = st.number_input("Pclass", value=3)
name = st.text_input("Name", value="Sharapova, Ms. Maria")
sex = st.selectbox("Sex", options=["male", "female"], index=1)
age = st.number_input("Age", value=24)
sibsp = st.number_input("SibSp", value=0)
parch = st.number_input("Parch", value=0)
ticket = st.text_input("Ticket", value="")
fare = st.number_input("Fare", value=112.0)
cabin = st.text_input("Cabin", value="")
embarked = st.selectbox("Embarked", options=["C", "Q", "S"], index=2)

# 添加預測按鈕
if st.button("Predict"):
    # 創建一個DataFrame來存儲輸入數據
    input_data = pd.DataFrame({
        "PassengerId": [passenger_id],
        "Pclass": [pclass],
        "Name": [name],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Ticket": [ticket],
        "Fare": [fare],
        "Cabin": [cabin],
        "Embarked": [embarked]
    })

    # 進行預測
    predictions = predict(input_data)
    if predictions['predictions'][1] == 1:
        st.image('survival-survive.jpg')
    # 顯示預測結果
    st.write(predictions)
