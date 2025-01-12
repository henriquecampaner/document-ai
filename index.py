import streamlit as st
import os
import faiss
import tempfile
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
  page_title="Talk to your documents 📚",
  page_icon="📚",
)
st.title("Talk to your documents 📚")

model_class = "hf_hub" # "hf_hub", "openai", "ollama"

def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
    llm = HuggingFaceEndpoint(
        repo_id = model,
        temperature = temperature, 
        max_new_tokens = 512,
        return_full_text = False,
        stop= ["<|eot_id|>"],
    )

    return llm

def model_openai(model = "gpt-4o-mini", temperature = 0.1):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
    )
    return llm

def model_ollama(model = "phi3", temperature = 0.1):
    llm = ChatOllama(model = model, temperature = temperature)
    return llm

def config_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_file_path)
        docs.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1000,
          chunk_overlap  = 200,
      )
    splits = text_splitter.split_documents(docs)

    # Create embeddings for the documents
    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")

    # Store the embeddings in a FAISS index
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorestore/db_faiss')

    # Config retriever
    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 3, "fetch_k": 4},
    )

    return retriever

def config_rag_chain(model_class, retriever):
  if model_class == "hf_hub":
        llm = model_hf_hub()
  elif model_class == "openai":
        llm = model_openai()
  elif model_class == "ollama":
        llm = model_ollama()

  if model_class.startswith("hf"):
      token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

  else:
      token_s, token_e = "", ""
  
  # Context Prompt
  context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
  context_q_system_prompt = token_s + context_q_system_prompt
  context_q_user_prompt = "Question: {input}" + token_e
  context_q_prompt = ChatPromptTemplate.from_messages([
      ("system", context_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", context_q_user_prompt),
  ])

  # Context Chain
  history_aware_retriever = create_history_aware_retriever(
      llm,
      retriever,
      prompt = context_q_prompt
  )

  # QA Prompt
  qa_prompt_template = """You are a helpful virtual assistant responding to general questions.
  Use the following retrieved pieces of context to answer the question.
  If you don't know the answer, just say you don't know. Keep the response concise.
  Respond in English. \n\n
  Question: {input} \n
  Context: {context}"""

  qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

  # LLM Chain
  qa_chain = create_stuff_documents_chain(
      llm,
      qa_prompt
  )
  rag_chain = create_retrieval_chain(
      history_aware_retriever,
      qa_chain
  )

  return rag_chain

uploads = st.sidebar.file_uploader(
    label="Upload a PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploads:
    st.write("Please upload a PDF file to start the conversation.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you today?")]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("ai"):
        st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("human"):
        st.write(message.content)

state = time.time()

user_query = st.chat_input("Type your question")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("human"):
        st.markdown(user_query)
    
    with st.chat_message("ai"):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)
        
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history,
        })

        response = result["answer"]

        st.write(response)

        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', "Page not found")

            ref = f":link: Source {idx}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)
    
    st.session_state.chat_history.append(AIMessage(content=response))

end = time.time()
print("Time taken: ", end - state)