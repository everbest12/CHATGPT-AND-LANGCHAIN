from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter( # separate text into chunks
    separator="\n",
    chunk_size = 200, #groups the text ino 200 characters
    chunk_overlap=0 #pulls the last 0 characters 
)

loader = TextLoader ("facts.txt") #split text into documents
docs = loader.load_and_split(
    text_splitter=  text_splitter
)
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)
 
results = db.similarity_search("english language")
for result in results:
    print("\n")
    print (result.page_content)
