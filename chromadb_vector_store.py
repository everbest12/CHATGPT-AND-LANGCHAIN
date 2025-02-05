from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
'''Whats Happening? This part brings in the tools (like ingredients in a recipe) needed to:1. Read text files. (2)Understand the meaning of words using AI (like how Google knows what you're looking for).
(3)Store the information smartly so it's easy to search. (4)Handle secret information (like API keys) securely.'''

embeddings = OpenAIEmbeddings() 
text_splitter = CharacterTextSplitter( # separate text into chunks
    separator="\n",
    chunk_size = 200, #groups the text ino 200 characters
    chunk_overlap=0 #pulls the last 0 characters 
)

loader = TextLoader ("facts.txt") #split text into documents
docs = loader.load_and_split( #

    text_splitter=  text_splitter
)
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)
 
results = db.similarity_search(" what is the interesting fact about the English language")
for result in results:
    print("\n")
    print (result.page_content)
    
