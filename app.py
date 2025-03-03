import streamlit as st
from search import rag_with_mistral
from pinecone import Pinecone

# streamlit app layout
def app():
    pc_key = st.secrets["PC_KEY"]
    hf_key = st.secrets['HF_KEY']
    pc_ind = st.secrets['PC_IND']

    pc = Pinecone(api_key=pc_key)
    index = pc.Index(pc_ind)

    st.title("LLM Question Answering App with Atheisism RAG Implementation")
    
    # Input section: User enters a question
    user_query = st.text_input("Enter your question:", "")

    if user_query:
        with st.spinner("Processing..."):
            
            normal_answer, rag_answer = rag_with_mistral(user_query, index, hf_key) 

        # Display results
        st.subheader("Normal LLM Answer:")
        st.write(normal_answer)  
        
        st.subheader("RAG Answer:")
        st.write(rag_answer) 

if __name__ == "__main__":
    
    app()
