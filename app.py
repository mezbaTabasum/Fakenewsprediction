import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

st.set_page_config(page_title="Fake News Chatbot", layout="centered")
st.title("🧠 Fake News Detection Chatbot")

@st.cache_data
def load_model_and_vectorizer():
    true_df = pd.read_csv("True.csv")
    false_df = pd.read_csv("Fake.csv")

    true_df['label'] = 1  # real
    false_df['label'] = 0  # fake

    df = pd.concat([True_df[['text', 'label']], Fake_df[['text', 'label']]])
    df.dropna(subset=['text'], inplace=True)

    X = df['text']
    y = df['label']

    vectorizer = tfidf_vectorizer(stop_words='english', max_df=0.7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_tfidf))

    return model, vectorizer, acc

model, vectorizer, accuracy = load_model_and_vectorizer()

st.info(f"✅ Model trained with {accuracy * 100:.2f}% accuracy")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Enter your news content here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Make prediction
    input_tfidf = vectorizer.transform([prompt])
    prediction = model.predict(input_tfidf)[0]

    response = "🟢 This news appears to be Real." if prediction == 1 else "🔴 This news appears to be Fake."

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
