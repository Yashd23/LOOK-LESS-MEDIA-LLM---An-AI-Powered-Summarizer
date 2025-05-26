import streamlit as st
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.api.types import EmbeddingFunction
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import textwrap
import os


NVIDIA_API_KEY = "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"
NVIDIA_MODEL = "meta/llama3-8b-instruct"  # Supported model from NVIDIA API Catalog
llm = ChatNVIDIA(model=NVIDIA_MODEL, api_key=NVIDIA_API_KEY)

# Custom embedding function to adapt NVIDIAEmbeddings to ChromaDB's interface
class CustomNVIDIAEmbeddingFunction(EmbeddingFunction):
    def __init__(self, nvidia_embeddings):
        self.nvidia_embeddings = nvidia_embeddings
    
    def __call__(self, input):
        # Adapting NVIDIAEmbeddings to ChromaDB's expected signature
        return self.nvidia_embeddings.embed_documents(input)

# Initialize NVIDIA embeddings (Cloud Setup)
try:
    nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
    embedding_function = CustomNVIDIAEmbeddingFunction(nvidia_embeddings)
except Exception as e:
    st.error(f"Failed to initialize embeddings: {str(e)}")
    st.stop()


# Initialize ChromaDB client for vector storage (local by default)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="multi_modal_content", embedding_function=embedding_function, get_or_create=True)

# Function to scrape text from a blog URL
def scrape_blog(blog_url):
    try:
        response = requests.get(blog_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "article"])
        blog_text = " ".join([element.get_text().strip() for element in text_elements])
        if not blog_text.strip():
            raise ValueError("No text content found in the blog/vidio/audio.")
        return blog_text
    except Exception as e:
        raise ValueError(f"Failed to scrape blog/vidio/audio from {blog_url}: {str(e)}")

# Function to split text into chunks for vector storage
def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to store content in ChromaDB
def store_in_vector_db(content, source, source_type):
    chunks = split_text(content)
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, metadatas=[{"source": source, "type": source_type} for _ in chunks], ids=chunk_ids)

# Function to generate summary using NVIDIA model
def generate_summary(content):
    prompt = f"""
    Summarize the following content in 100-150 words, capturing the main ideas and key points:

    {content}

    Ensure the summary is concise, clear, and retains the core message.
    """
    response = llm.invoke(prompt)
    return response.content

# Function to generate insights using NVIDIA model
def generate_insights(content):
    prompt = f"""
    Analyze the following content and provide 3-5 key insights or takeaways. Focus on unique perspectives, trends, or implications:

    {content}

    Format the insights as a bullet-point list.
    """
    response = llm.invoke(prompt)
    return response.content

# Function for Q&A using RAG
def answer_question(question):
    results = collection.query(query_texts=[question], n_results=3)
    retrieved_content = " ".join(results["documents"][0])
    prompt = f"""
    Based on the following context, answer the question: {question}

    Context: {retrieved_content}

    Provide a clear and accurate answer.
    """
    response = llm.invoke(prompt)
    return response.content

# Streamlit app
def main():
    st.set_page_config(page_title="Summarizer RAG System", layout="wide")
    st.title("📝 Look Less Media - An AI-Powered Summarizer and Q&A")

    # Blog URL input
    blog_url = st.text_input("Enter Url:", placeholder="https://www.example.com/blog/vidio/audio")
    
    # Process button
    if st.button("Process"):
        if not blog_url:
            st.error("Please enter a valid URL.")
            return
        
        with st.spinner("Processing ...."):
            try:
                blog_text = scrape_blog(blog_url)
                store_in_vector_db(blog_text, blog_url,"blog")
                
                # Generate and display summary
                with st.spinner("Generating summary..."):
                    summary = generate_summary(blog_text)
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0;">
                            <h3 style="text-align: center; color: #4CAF50;">Summary</h3>
                            <p style="line-height: 1.6;">{textwrap.fill(summary, width=80)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Generate and display insights
                with st.spinner("Generating insights..."):
                    insights = generate_insights(blog_text)
                    st.subheader("Insights")
                    st.markdown(insights)
                
                # Store blog text in session state for Q&A
                st.session_state.blog_text = blog_text
            except ValueError as e:
                st.error(f"Error processing : {str(e)}")
                return

    # Interactive Q&A
    st.subheader("Interactive Q&A")
    if "blog_text" in st.session_state:
        question = st.text_input("Ask a question about the context:", key="qa_input")
        if st.button("Submit Question"):
            if not question.strip():
                st.warning("Please enter a valid question.")
            else:
                try:
                    with st.spinner("Generating answer..."):
                        answer = answer_question(question)
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")
    else:
        st.info("Process a  URL to enable Q&A.")

if __name__ == "__main__":
    main()