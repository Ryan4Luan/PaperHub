from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.vectorstores import Pinecone
from datasets import load_dataset
from pinecone import PodSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from dotenv import load_dotenv  # Only if using python-dotenv

import datasets
import pandas as pd
import time
import requests
import os

# Load environment variables from .env file (if using python-dotenv)
load_dotenv()


class MedChatbot:
    def __init__(self):
        # Initialize API_KEYS
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # self.data_path = "jamescalam/llama-2-arxiv-papers-chunked"
        self.data_path = "../assets/data/medical_articles.json"
        self.embedding_model = "text-embedding-ada-002"
        self.embed_model = OpenAIEmbeddings(model=self.embedding_model, api_key=self.OPENAI_API_KEY)
        self.spec = PodSpec(
            environment=self.PINECONE_ENVIRONMENT, pod_type="starter"
        )
        self.index_name = 'medical-intelligence-rag'

    def load_data(self):
        """
        Load data from a specified file path.
        """
        # Temporarily disable SSL verification
        requests.packages.urllib3.disable_warnings()
        datasets.utils.VerificationMode = False

        # dataset = load_dataset(
        #     self.data_path,
        #     split="train"
        # )
        # print(dataset.to_pandas())
        # data = dataset.to_pandas()

        # Read the JSON file
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = pd.read_json(file)

        # Convert the 'articles' list in the JSON data to a DataFrame
        df = pd.DataFrame(data['articles'])
        return df

    def load_vectorstore(self):
        from pinecone import Pinecone
        return Pinecone(api_key=self.PINECONE_API_KEY)

    def create_index(self, vector_store, data, batch_size=100):
        """
        Create an index for the loaded data.
        """
        existing_indexes = [
            index_info["name"] for index_info in vector_store.list_indexes()
        ]

        # check if index already exists (it shouldn't if this is first time)
        if self.index_name not in existing_indexes:
            # if does not exist, create index
            vector_store.create_index(
                self.index_name,
                dimension=1536,  # dimensionality of ada 002
                metric='dotproduct',
                spec=self.spec
            )
            # wait for index to be initialized
            while not vector_store.describe_index(self.index_name).status['ready']:
                time.sleep(1)

        # connect to index
        index = vector_store.Index(self.index_name)
        time.sleep(1)

        dataset = data  # this makes it easier to iterate over the dataset

        print("Creating embeddings for data ...\n")
        for i in tqdm(range(0, len(dataset), batch_size)):
            i_end = min(len(dataset), i + batch_size)
            # get batch of data
            batch = dataset.iloc[i:i_end]

            # generate unique ids for each chunk
            ids = [f"{x['articles']['id']}-{x['articles']['chunk-id']}" for i, x in batch.iterrows()]
            # get text to embed
            texts = [x['articles']['chunk'] for _, x in batch.iterrows()]
            # embed text
            embeds = self.embed_model.embed_documents(texts)
            # get metadata to store in Pinecone
            metadata = [
                {'text': x['articles']['chunk'],
                 'source': x['articles']['source'],
                 'title': x['articles']['title'],
                 'authors': x['articles']['authors'],
                 'journal_ref': x['articles']['journal_ref'],
                 'published': x['articles']['published']
                 } for i, x in batch.iterrows()
            ]
            # add to Pinecone
            index.upsert(vectors=zip(ids, embeds, metadata))
            print(index.describe_index_stats())
        return index

    def get_index(self, vectorstore):
        # connect to index
        return vectorstore.Index(self.index_name)

    def augment_prompt(self, query, k, vectorstore):
        # get top 3 results from knowledge base
        results = vectorstore.similarity_search(query, k)
        # get the text from the results
        source_knowledge = "\n".join([x.page_content for x in results])
        # feed into an augmented prompt
        augmented_prompt = f"""Using the contexts below, answer the query.
        
        Contexts:
        {source_knowledge}

        Query: {query}"""
        return augmented_prompt, results

    def query(self, index, query, k, text_field="text", chat_model="gpt-3.5-turbo"):
        # initialize the vector store object
        from langchain.vectorstores import Pinecone
        vectorstore = Pinecone(
            index, self.embed_model.embed_query, text_field
        )
        chat = ChatOpenAI(
            openai_api_key=self.OPENAI_API_KEY,
            model=chat_model
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hi AI, how are you today?"),
            AIMessage(content="I'm great thank you. How can I help you?")
        ]
        augment_prompt, retriever_results = self.augment_prompt(query, k, vectorstore)

        prompt = HumanMessage(
            content=augment_prompt
            # content=query
        )
        res = chat(messages + [prompt])
        return res, retriever_results


if __name__ == '__main__':
    chatbot = MedChatbot()
    # data = chatbot.load_data()
    vc = chatbot.load_vectorstore()
    # index = chatbot.create_index(vc, data, batch_size=100)
    index = chatbot.get_index(vc)