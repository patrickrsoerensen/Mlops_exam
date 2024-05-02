import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import plotly.express as px
import pickle

# Load your trained SVM model
with open("svm_classifier.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the dataset of news articles
news_df = pd.read_csv("rating.csv")

# Function to preprocess and vectorize the input title
def preprocess_and_vectorize(title):
    preprocessed_title = vectorizer.transform([title])
    return preprocessed_title

# Function to predict sentiment
def predict_sentiment(title):
    preprocessed_title = preprocess_and_vectorize(title)
    sentiment_prediction = svm_classifier.predict(preprocessed_title)
    return sentiment_prediction

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    st.title("News Sentiment Analysis")

    # Dropdown to select news source
    selected_source = st.selectbox("Select a news source:", news_df["source_name"].unique())

    # Filter articles based on selected source
    source_articles = news_df[news_df["source_name"] == selected_source]

    if not source_articles.empty:
        # Create two columns layout
        col1, padding, col2 = st.columns((4,1,4))


        # Display top 10 newest articles from selected source on the left column
        with col1:
            st.write("### Top 10 Newest Articles from", selected_source)
            for index, row in source_articles.head(10).iterrows():
                st.markdown("---")
                st.subheader(row["title"])
                st.write("Published at:", row["published_at"])
                st.image(row["url_to_image"], caption="Image for the article", use_column_width=True)
                st.write("URL:", row["url"])
                sentiment = predict_sentiment(row["title"])
                st.write("Sentiment:", sentiment[0])

        # Create a pie chart showing sentiment distribution on the right column
        with col2:
            st.write("### Sentiment Distribution")
            sentiment_counts = source_articles["title"].apply(predict_sentiment).value_counts()
            pie_chart = px.pie(values=sentiment_counts, names=sentiment_counts.index)
            st.plotly_chart(pie_chart, use_container_width=True)

    else:
        st.write("No articles found for the selected source.")

if __name__ == "__main__":
    main()
