from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from query import *
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv  # Only if using python-dotenv

# Load environment variables from .env file (if using python-dotenv)
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = "text-embedding-ada-002"
embedding_model = OpenAIEmbeddings(model=embedding_model, api_key=OPENAI_API_KEY)


def evaluate_with_embeddings(true_texts, predicted_texts, embedding_model):
    """
    Evaluates the similarity between true and predicted answers using embeddings and cosine similarity.

    Parameters:
    - true_texts: list of true answer texts
    - predicted_texts: list of predicted answer texts by the model
    - embedding_model: an instance of OpenAIEmbeddings or similar, with a method to generate embeddings

    Returns:
    - similarities: a list of cosine similarity scores between all true and predicted answer pairs
    """
    # Generate embeddings for both true and predicted texts
    true_embeddings = np.array([embedding_model.embed_query(text) for text in true_texts])
    predicted_embeddings = np.array([embedding_model.embed_query(text) for text in predicted_texts])

    # Calculate cosine similarity for each pair of true and predicted embeddings
    similarities = []
    for true_emb, pred_emb in zip(true_embeddings, predicted_embeddings):
        # Ensure embeddings are reshaped to 2D arrays for cosine_similarity function
        similarity = cosine_similarity(true_emb.reshape(1, -1), pred_emb.reshape(1, -1))[0][0]
        similarities.append(similarity)

    return similarities


def plot_eval():
    data = {
        'Q1': [0.972, 0.981, 0.972],
        'Q2': [0.977, 0.952, 0.948],
        'Q3': [0.982, 0.965, 0.954],
        'Q4': [0.977, 0.963, 0.934],
        'Q5': [0.980, 0.958, 0.952],
        'Q6': [0.991, 0.975, 0.972],
        'Q7': [0.975, 0.971, 0.969],
        'Q8': [0.990, 0.929, 0.952],
        'Q9': [0.981, 0.967, 0.960],
    }
    df = pd.DataFrame(data, index=['RAG', 'GPT-3', 'GPT-4'])

    # Transpose the DataFrame for plotting
    df_t = df.T

    # Plotting each model's scores
    plt.figure(figsize=(10, 5))
    plt.plot(df_t['RAG'], label='RAG', color='blue', marker='o')
    plt.plot(df_t['GPT-3'], label='GPT-3', color='green', marker='o')
    plt.plot(df_t['GPT-4'], label='GPT-4', color='red', marker='o')

    plt.title('Model Evaluation Scores by Question')
    plt.xlabel('Question Number')
    plt.ylabel('Embedding Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of ylabel
    plt.show()


if __name__ == '__main__':
    chatbot = MedChatbot()
    # data = chatbot.load_data()
    vc = chatbot.load_vectorstore()
    # index = chatbot.create_index(vc, data, batch_size=100)
    index = chatbot.get_index(vc)

    # Load questions from CSV file
    questions_df = pd.read_csv('../assets/data/data_eval.csv')

    pred_answers_list = []
    answers_lsit = []

    for idx, row in questions_df.iterrows():
        question = row['question']
        answer = row['answer']
        pre_answer, _ = chatbot.query(index, question, k=3, text_field="text", chat_model="gpt-4")

        # Append the answer to the answers DataFrame
        pred_answers_list.append(pre_answer.content)
        answers_lsit.append(answer)

    print(evaluate_with_embeddings(answers_lsit, pred_answers_list, embedding_model))