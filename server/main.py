from dotenv import load_dotenv
_ = load_dotenv('../.env')

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

from langchain.docstore.document import Document
from pypdf import PdfReader

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io

db = None
memory = None
summary = None

def pdf_to_document(reader):
    text = '\n'.join([page.extract_text() for page in reader.pages])
    doc = Document(page_content=text, metadata={"source": "NA"})
    return doc

def summarize_doc(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    
    documents = text_splitter.split_documents([doc])

    llm = ChatOpenAI(temperature=0)
    
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify generally what the document is about
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # Reduce
    reduce_template = """The following is an unordered set of summaries based on parts of a document:
    {docs}
    Take these and write a summary of the document.
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )
    
    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    docs_to_summarize = documents[:5]

    summary = map_reduce_chain.invoke(docs_to_summarize)
    return summary['output_text']

def generate_query(user_input, memory):
    template = '''You will be provided with an initial prompt and some chat history. If the initial prompt is too vague, add more context to it with the help of the chat history.
    be sure that the new prompt is concise in one sentence, to the point, and highly relevant to be used in querying information from an insightful text document.
    Input Prompt: {user_input}
    History: {chat_history}
    New Prompt:'''

    prompt = PromptTemplate.from_template(template)

    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt,
    )

    query = chain.invoke({"user_input": user_input, "chat_history": memory.load_memory_variables({})["chat_history"]})
    return query["text"]

def generate_context(db, query):
    context = '\n'.join([doc.page_content for doc in db.similarity_search(query, k=4)])
    return context
    
def generate_final_response(context, user_input, memory):
    template = """Use the following chat history and pieces of context to give a helpful answer at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible and don't repeat yourself.
    
    History: {chat_history}
    
    Context: {context}

    Question: {user_input}
    
    Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt
    )

    response = chain.invoke({"chat_history": memory.load_memory_variables({})["chat_history"], "context": context, "user_input": user_input})
    return response

def prompt_db_doc(db, user_input, memory, verbose=0):
    query = generate_query(user_input, memory)
    context = generate_context(db, query)
    response = generate_final_response(context, user_input, memory)
    response_text = response["text"]
    memory.save_context({"input": user_input}, {"output": response_text})
    if verbose == 1:
        print(f"Query: {query}\n")
        print(f"Context: {context}\n")
        print(f"Invoke: {response}\n")
    return response_text

def initialize_memory():
    global memory
    global summary
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
    memory.save_context({"input": "What is this document about?"}, {"output": summary})

def upload_to_docbot(reader):
    doc = pdf_to_document(reader)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    documents = text_splitter.split_documents([doc])

    collection_name = "default_collection"
    
    global db
    if db:
        db.delete_collection()
    db = Chroma.from_documents(documents, 
                            embedding=OpenAIEmbeddings(),
                            collection_name=collection_name)

    global summary
    summary = summarize_doc(doc)

    global memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
    memory.save_context({"input": "What is this document about?"}, {"output": summary})




# upload_to_docbot('./dev/docs/WaltDisney.pdf')
# print("summary: ", summary, "\n")

# upload_to_docbot('./dev/docs/relu.pdf')
# print("summary: ", summary, "\n")


# while True:
#     user_in = input("\ninput: ")
#     response = prompt_db_doc(db, user_in, memory)
#     print("\nresponse: ", response)

def base64_to_pdf_reader(base64_data):
    buffer=base64.b64decode(base64_data)
    f=io.BytesIO(buffer)
    reader = PdfReader(f)
    return reader



# SETUP API

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTE API

@app.get("/")
async def root(input:str = None):
    return {"message": "Hello World!"}

class Item(BaseModel):
    data: str = None

@app.post("/begin_chat")
async def begin_chat(item:Item = None):
    if not item or not item.data:
        raise HTTPException(status_code=400, detail="PDF failed to send.")
    else:
        reader = base64_to_pdf_reader(item.data)
        upload_to_docbot(reader)
        return {"message": summary}

@app.post("/send_prompt")
async def send_prompt(item:Item = None):
    if not item or not item.data:
        return {"message": "Please enter an input"}
    else:
        response = prompt_db_doc(db, item.data, memory)
        return {"message": response}
    
@app.get("/reset")
async def reset():
    initialize_memory()
    return {"message": "Memory initialized!"}


@app.post("/pdf_test")
async def pdf_test(item: Item):
    data_str = item.data
    buffer=base64.b64decode(data_str)
    f=io.BytesIO(buffer)
    reader = PdfReader(f)
    page = reader.pages[0]
    page_txt = page.extract_text()
    return {"message": page_txt}