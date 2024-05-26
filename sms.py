import streamlit as st
import pickle


model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('TfidfVectorizer.pkl','rb'))

st.title("Email/spam Classifier")
msg=st.text_input("Enter the Text:")
if st.button("predict"):
    data=[msg]
    vt=cv.transform(data).toarray()
    prition=model.predict(vt)
    rsult=prition[0]
    if rsult == 1:
        st.error("this is a spam mail")
    else:
        st.error("this is a Ham mail")

