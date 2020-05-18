import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')

import streamlit as st
from Question_Answering import answer_question

st.title("Question Answering App")
st.subheader("Seq2Seq with attention Demo")
corpus_data = st.text_area("Enter your passage here")
question = st.text_input("Question")
result_ans = answer_question(question, corpus_data)

if(st.button("Submit")):
    st.success(result_ans)

