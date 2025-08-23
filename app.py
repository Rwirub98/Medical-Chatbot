from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embeddings = download_embeddings()

# Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever + Chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    # Try to fetch msg safely
    msg = request.form.get("msg")
    if not msg and request.is_json:
        data = request.get_json(silent=True)
        msg = data.get("msg") if data else None

    print("User:", msg)

    if not msg:
        return "Sorry, I didn’t get your message."

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I don’t know how to answer that.")
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print("Error:", e)
        return "Oops! Something went wrong. Please try again."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
