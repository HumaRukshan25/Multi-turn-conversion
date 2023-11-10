import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from jinja2 import Template
from langchain.llms import HuggingFaceHub
import sqlite3
from sqlite3 import Error

def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def create_connection():
    connection = None
    try:
        connection = sqlite3.connect("conversation_history.db")
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"Error: {e}")
    return connection

def create_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT,
        bot_message TEXT
    );
    """
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        print("Table created successfully")
    except Error as e:
        print(f"Error: {e}")

def insert_message(connection, user_message, bot_message):
    insert_query = "INSERT INTO conversation_history (user_message, bot_message) VALUES (?, ?);"
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (user_message, bot_message))
        connection.commit()
        print("Message inserted successfully")
    except Error as e:
        print(f"Error: {e}")

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    connection = create_connection()
    create_table(connection)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            template = Template("<div class='user'>{{MSG}}</div>")
            st.write(template.render(MSG=message.content), unsafe_allow_html=True)
            insert_message(connection, user_message=message.content, bot_message=None)
        else:
            template = Template("<div class='bot'>{{MSG}}</div>")
            st.write(template.render(MSG=message.content), unsafe_allow_html=True)
            insert_message(connection, user_message=None, bot_message=message.content)

    connection.close()

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        process_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = create_vectorstore(text_chunks)
                st.session_state.conversation = create_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
