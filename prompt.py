from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import langchain
langchain.debug = True

load_dotenv()
chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function = embeddings
) 
retriever = db.as_retriever()
'''retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma = db'''
    
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
    
)
result = chain.run(" what is the interesting fact about the  english language")
print(result)