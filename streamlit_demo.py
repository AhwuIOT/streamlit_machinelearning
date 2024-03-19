import streamlit as st
import pandas as pd
import pickle

# 加載模型和pipeline
model = pickle.load(open('model.pkl', 'rb')) 
pipeline = pickle.load(open('pipeline.pkl', 'rb'))
def predict(json_data):
    # 將JSON數據轉換為DataFrame
    if isinstance(json_data['PassengerId'], list):
        entry = pd.DataFrame(json_data)
    else:
        entry = pd.DataFrame([json_data])
    
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

# 獲取JSON輸入
json_input = st.text_area("Enter JSON data")

# 添加預測按鈕
if st.button("Predict"):
    # 解析JSON輸入
    input_data = eval(json_input)
    
    # 進行預測
    predictions = predict(input_data)
    
    # 顯示預測結果
    st.write(predictions)