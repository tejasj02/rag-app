import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st

def query_pinecone(index, query_vector, top_k=1000):
    response = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_values=True,  
        include_metadata=True  
    )

    results = [(match.values, match.metadata["sentence"]) for match in response.matches if "sentence" in match.metadata]
    return results

# calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# generate context
def rag_with_mistral(query, index, hf_key, model_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode([query])[0]
    
    retrieved_results = query_pinecone(index, query_embedding, top_k=1000)
    
    similarities = [cosine_similarity(query_embedding, embedding) for embedding, _ in retrieved_results]
    
    top_n = 5
    top_indices = np.argsort(similarities)[-top_n:]
    top_context = [retrieved_results[i][1] for i in top_indices] 
    
    context = "\n".join(top_context)
    prompt = f"Context:\n{context}\n\nQuery: {query}"

    return query_mistral_hub(query, prompt, hf_key)

# prompt the LLM
def query_mistral_hub(prompt, prompt_context, api_token):
    
    client = InferenceClient(token=api_token)

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    formatted_prompt_context = f"<s>[INST] {prompt_context} [/INST]"
    
    response = client.text_generation(
        formatted_prompt,
        model=model_id,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response_context = client.text_generation(
        formatted_prompt_context,
        model=model_id,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response_text = response.replace(formatted_prompt, "").strip()
    response_text_context = response_context.replace(formatted_prompt_context, "").strip()
    
    return response_text, response_text_context

if __name__ == "__main__":
    pc_key = st.secrets["PC_KEY"]
    hf_key = st.secrets['HF_KEY']
    pc_ind = st.secrets['PC_IND']
    pc = Pinecone(api_key=pc_key)
    index = pc.Index(pc_ind)
    query = "Are atheists morally insane?"
    answer = rag_with_mistral(query, index, hf_key)
    print(answer)
