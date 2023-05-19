import os
import sys

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

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


def get_sys_args():
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])


if __name__ == "__main__":
    load_file(sys.argv[1])
