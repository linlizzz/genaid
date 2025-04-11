"""
# Embedding and Vector Database
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import TextLoader

persist_directory = "data/chroma_langchain_db_test"
model_name = "LumiOpen/poro-34b-chat"



def get_embedding():
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings   

def create_db():
    
    loader = TextLoader("data/clinical_guidelines.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, embedding=get_embedding(), persist_directory=persist_directory)
    
    vectordb.persist()

# create_db()


# Retrieval QA
def retrieval_qa():

    model = OllamaLLM(model=model_name)

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=get_embedding())
    
    qa = VectorDBQA.from_chain_type(llm=model, chain_type="stuff", vectorstore=vectordb)
    
    result = qa.run(query)
    return result


query = "What is the clinical care guideline for pneumonia?"

response = retrieval_qa(query)
print(response)
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub


# Initialize the model
model = HuggingFaceHub(
    hub_id="LumiOpen/poro-34b-chat", 
    model_kwargs={"temperature": 0.2}
)

# Create a prompt template
template = PromptTemplate(
    input_variables=["topic"],
    template="Kirjoita tarina {topic}",
)

# Build the chain using the template and model
llm_chain = LLMChain(
    llm=model,
    prompt=template,
)

# Execute the chain
story_response = llm_chain.run("ystävyydestä")
print(story_response)

