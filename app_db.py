import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp
import mysql.connector
from mysql.connector import Error
import uuid
import datetime
import os

def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'chat_history_db'),
            user=os.getenv('DB_USER', 'your_username'),
            password=os.getenv('DB_PASSWORD', 'your_password')
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

def initialize_db():
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                session_name VARCHAR(255),
                created_at DATETIME
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255),
                timestamp DATETIME,
                sender VARCHAR(10),
                message TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
            )
        """)
        connection.commit()
        cursor.close()
        connection.close()

def create_new_session(session_name=None):
    session_id = str(uuid.uuid4())
    if not session_name:
        session_name = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO chat_sessions (session_id, session_name, created_at)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (
            session_id,
            session_name,
            datetime.datetime.now()
        ))
        connection.commit()
        cursor.close()
        connection.close()
    return session_id

def get_previous_sessions():
    connection = create_connection()
    sessions = []
    if connection:
        cursor = connection.cursor(dictionary=True)
        select_query = """
            SELECT session_id, session_name FROM chat_sessions
            ORDER BY created_at DESC
        """
        cursor.execute(select_query)
        records = cursor.fetchall()
        for record in records:
            sessions.append({
                'session_id': record['session_id'],
                'session_name': record['session_name']
            })
        cursor.close()
        connection.close()
    return sessions

def load_chat_history_for_session(session_id):
    connection = create_connection()
    messages = []
    if connection:
        cursor = connection.cursor(dictionary=True)
        select_query = """
            SELECT sender, message FROM chat_history
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """
        cursor.execute(select_query, (session_id,))
        records = cursor.fetchall()
        for record in records:
            messages.append({
                'sender': record['sender'],
                'message': record['message']
            })
        cursor.close()
        connection.close()
    return messages

def save_message_to_db(session_id, sender, message):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO chat_history (session_id, timestamp, sender, message)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            session_id,
            datetime.datetime.now(),
            sender,
            message
        ))
        connection.commit()
        cursor.close()
        connection.close()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="lmsys/vicuna-7b-v1.3",
    #     task="text-generation",
    #     model_kwargs={"temperature": 0.01},
    # )
    # llm = LlamaCpp(
    #    model_path="models/llama-2-7b-chat.ggmlv3.q4_1.bin",  n_ctx=1024, n_batch=512)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def handle_userinput():
    # check if the conversation has been exited
    if st.session_state.get('exited', False):
        st.write("The conversation has ended. Click 'New Chat' to start a new session.")
        return

    user_message = st.chat_input("Ask questions about your documents:")
    if user_message:
        if user_message.strip().lower() == 'exit':
            # Handle exit command
            st.write("**Conversation ended.**")
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.exited = True
            return
        else:
            if st.session_state.conversation is None:
                st.error("Please upload and process documents first.")
            else:
                # save user message to db
                save_message_to_db(st.session_state.session_id, 'user', user_message)

                response = st.session_state.conversation({'question': user_message})
                st.session_state.chat_history = response['chat_history']

                # get the GPT's reply
                assistant_message = st.session_state.chat_history[-1].content

                # save GPT message to database
                save_message_to_db(st.session_state.session_id, 'assistant', assistant_message)

                # display the current chat history
                for message in st.session_state.chat_history:
                    if message.type == 'human':
                        with st.chat_message("user"):
                            st.write(message.content)
                    else:
                        with st.chat_message("assistant"):
                            st.write(message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    initialize_db()

    # initialize session_state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "exited" not in st.session_state:
        st.session_state.exited = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # No session started yet
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None

    st.header("Chat with PDFs :robot_face:")

    # sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Start a new session if not already started
                    if st.session_state.session_id is None:
                        st.session_state.session_id = create_new_session()
                    
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

                    st.success("Processing complete! You can now ask questions.")
            else:
                st.error("Please upload at least one PDF document.")

        st.markdown("---")
        st.subheader("Previous Chat Sessions")
        sessions = get_previous_sessions()
        if sessions:
            session_names = [s['session_name'] for s in sessions]
            session_ids = [s['session_id'] for s in sessions]
            selected_session_name = st.selectbox(
                "Select a session to view", [""] + session_names)
            if selected_session_name:
                # get the session_id for the selected session
                index = session_names.index(selected_session_name)
                selected_session_id = session_ids[index]
                st.session_state.selected_session_id = selected_session_id

                # load messages for the selected session
                messages = load_chat_history_for_session(selected_session_id)

                # display messages in an expander
                with st.expander(f"Chat History for {selected_session_name}"):
                    for message in messages:
                        if message['sender'] == 'user':
                            st.write(f"**User:** {message['message']}")
                        else:
                            st.write(f"**Assistant:** {message['message']}")
        else:
            st.write("No previous chat sessions.")

        st.markdown("---")
        st.subheader("Actions")
        if st.button("New Chat"):
            # Create a new session
            st.session_state.session_id = create_new_session()
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.exited = False
            st.session_state.selected_session_id = None
            st.success("Started a new chat session. Please process documents to begin.")
            
    # handle current chat
    handle_userinput()

if __name__ == '__main__':
    main()

