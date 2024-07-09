import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Perform stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model from disk
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set up the Streamlit interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # 3. Convert sparse matrix to dense
    vector_input_dense = vector_input.toarray()
    # 4. Predict the label
    result = model.predict(vector_input_dense)[0]
    # 5. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
