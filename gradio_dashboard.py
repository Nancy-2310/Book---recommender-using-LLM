import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Create documents directly from DataFrame instead of loading from file
documents = []
for _, row in books.iterrows():
    content = f"{row['isbn13']} {row['description']}"
    documents.append(Document(page_content=content))

# Create the vector database using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(float(rec.page_content.strip('"').split()[0])) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Only sort by emotion if the columns exist
    if tone == "Happy" and "joy" in book_recs.columns:
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising" and "surprise" in book_recs.columns:
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry" and "anger" in book_recs.columns:
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful" and "fear" in book_recs.columns:
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad" and "sadness" in book_recs.columns:
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Fix: Filter out NaN values before sorting
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())

# Only include emotion tones if the emotion columns exist
emotion_columns = ["joy", "surprise", "anger", "fear", "sadness"]
emotion_labels = ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
available_emotions = [label for col, label in zip(emotion_columns, emotion_labels) if col in books.columns]
tones = ["All"] + available_emotions

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
