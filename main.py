import os
import sys

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def load_file(pdf_url):
    loader = PyPDFLoader("https://arxiv.org/pdf/2304.07327v1.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # チャンクの文字数
        chunk_overlap=50,  # チャンクオーバーラップの文字数。0の方がチャットボットの性能は良さそう
    )
    documents = text_splitter.split_documents(docs)
    print(documents[:3])

    return documents


def build_qa_engine(documents, embed_model):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    # db = Chroma.from_documents(documents, embeddings, persist_directory='embed_db')
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    # db.persist()

    # retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})
    retriever = db.as_retriever()

    prompt_template = """
    Use the following context to answer the question. 
    If you can't find the answer in the provided context, just answer with 'I cannot find the answer in the provided context.', don't try to make up an answer.
    Context is delimited with triple dashes. Question is delimited by triple backticks.
    
    Context: ---{context}---
    
    Question: ```{question}```
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa_machine = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        retriever=retriever,
        combine_docs_chain_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    return qa_machine


def get_sys_args():
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])


if __name__ == "__main__":
    documents = load_file(sys.argv[1])
    question = sys.argv[2]
    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qa_machine = build_qa_engine(documents, embed_model)

    response = qa_machine({"question": question, 'chat_history':''})

    print(response)
