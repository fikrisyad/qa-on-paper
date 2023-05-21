import os
import sys
import requests
import io

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def load_file(pdf_url):
    try:
        response = requests.get(pdf_url)

        if response.status_code == 200:
            with open('file/temp.pdf', 'wb') as f:
                f.write(response.content)

            loader = PyPDFLoader('file/temp.pdf')
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=10,
            )
            docs = text_splitter.split_documents(docs)
            return docs
        else:
            raise ConnectionError('File download failed')
    except ConnectionError:
        print('File download failed')


def build_qa_engine(documents, embed_model='openai'):
    if embed_model != 'openai':
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    else:
        embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    # db.persist()

    # retriever = db.as_retriever()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})


    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in the same language as the question:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}


    qa_machine = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        retriever=retriever,
        chain_type='stuff',
        chain_type_kwargs=chain_type_kwargs,
        # return_source_documents=True
    )

    return qa_machine


if __name__ == "__main__":
    documents = load_file(sys.argv[1])
    question = sys.argv[2]
    embed_model = "openai"
    # embed_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    qa = build_qa_engine(documents, embed_model)

    response = qa.run(question)
    print(response)
    # response = qa({"query": question})

    # print(f"{response['result']}\nSource:{response['source_documents']}")
