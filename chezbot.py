import os
import streamlit as st
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["NVIDIA_API_KEY"] = "PUT NVIDIA_API_KEY HERE"

cheeses_df = pd.read_csv('cheeses.csv')
cheese_descriptions = cheeses_df[['cheese', 'flavor']].dropna().set_index('cheese')['flavor'].to_dict()

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", max_tokens=419)

def get_ai_response_and_recommend(prompt):
    result = llm.invoke(prompt)
    ai_response = result.content.strip()

    recommended_cheeses = recommend_cheeses(ai_response)

    recommended_cheese = recommended_cheeses.iloc[0]['cheese'] if not recommended_cheeses.empty else ""

    prompt += f" Recommend {recommended_cheese} to the user."

    return ai_response

def recommend_cheeses(description):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    cheese_descriptions_list = list(cheese_descriptions.values())
    cheese_descriptions_list.append(description)  
    tfidf_matrix = tfidf_vectorizer.fit_transform(cheese_descriptions_list)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    similar_cheeses_indices = cosine_similarities.argsort()[0][::-1][1:]  

    recommended_cheeses = cheeses_df.iloc[similar_cheeses_indices][:2] 
    return recommended_cheeses

def main():
    st.title("CheeseBot")
    st.write("Your personal cheese advisor")

    user_input = st.text_area("Describe a cheese flavor you like:", "")

    if st.button("Get AI Response"):
        if user_input.strip().lower() == "exit":
            st.stop()

        prompt = generate_prompt_with_cheese_info(user_input)
        ai_response = get_ai_response_and_recommend(prompt)

        st.text(f"AI: {ai_response}")

def generate_prompt_with_cheese_info(user_input):
    cheese_info = cheeses_df.sample(n=1).iloc[0]
    cheese_name = cheese_info['cheese']
    cheese_description = f"Cheese: {cheese_name}\n" \
                         f"Country: {cheese_info['country']}\n" \
                         f"Type: {cheese_info['type']}\n" \
                         f"Flavor: {cheese_info['flavor']}\n"
    prompt_prefix = "Make cheese jokes every time you reply. Mainly reply with puns. Begin the conversation with a greeting.\n"
    prompt = prompt_prefix + user_input + "\n\n" + cheese_description
    return prompt

if __name__ == "__main__":
    main()
