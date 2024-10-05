import os
#import magic_bin as magic
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message

# Load environment variables
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
os.environ["GOOGLE_API_KEY"]='AIzaSyDTdH68aWCl5JWvPSF22QocrG_Cn7XrZM4'


@st.cache_data
def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embeddings_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = doc_preprocessing()
    db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://ibrahimkamran490/text_embedding")
    db = DeepLake(
        dataset_path=f"hub://ibrahimkamran490/text_embedding",
        read_only=True,
        embedding_function=embeddings,
    )
    return db

@st.cache_resource
def search_db():
    db = embeddings_store()
    retriever = db.as_retriever()
    #retriever.search_kwargs['distance_metric'] = 'cos'
    #retriever.search_kwargs['fetch_k'] = 100
    #retriever.search_kwargs['maximal_marginal_relevance'] = True
    #retriever.search_kwargs['k'] = 10
    model = ChatGroq(model="llama3-70b-8192")
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa

qa = search_db()

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    # Initialize Streamlit app with a title
    st.title("LLM Powered Chatbot")

    # Get user input from text input
    user_input = st.text_input("", key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        output = qa({'query': user_input})
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()
