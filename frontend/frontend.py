import streamlit as st
import pickle
from PIL import Image
import requests

# url_link="http://127.0.0.1:8000/predict"  #when manually check streamlit 
url_link="http://fastapi:8000/predict"


page=st.sidebar.radio("Go to",["Home","CNN Toxic Comment Classifier"])
if page=="Home":
    st.header("Toxic Comment Classifier")
    image=Image.open("C:\\Users\\Dell\\OneDrive - Havells\\Downloads\\Toxic_comment.png")
    st.image(image,use_container_width=True)
elif page=="CNN Toxic Comment Classifier":
    st.title("CNN Toxic Comment Classifier")
    
    text_input = st.text_area("Enter a comment")
    
    if st.button("Predict"):
        with st.spinner("Thinking...."):
            
            input_data={"Comment_text":text_input}
            
            try:
                response=requests.post(url_link,json=input_data)
                if response.status_code in [200,201,202]:
                    output=response.json()
                    st.write("Response:",output['result'])
                else:
                    st.warning(f"{response.status_code}->{response.text}")
            except Exception as e:
                st.write(e)
    
    

