from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Custom template for the conversation chain
custom_template = """You are a helpful AI assistant specializing in Computer Organization and Architecture. Use the following pieces of context and your general knowledge to answer the question. If the context doesn't contain relevant information, rely on your general knowledge to provide an accurate response.

Context: {context}

Current conversation:
{chat_history}

Question: {question}

Please format your response using proper markdown:
- Use **bold** for emphasis
- Use proper list formatting with dashes or numbers
- Use `code blocks` for code or technical terms
- Use > for quotes
- Use ### for headers where appropriate
- Use proper spacing between paragraphs
- Use tables when presenting structured information

Answer: """

# Initialize conversation chain
def initialize_chain():
    # Get API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Load pre-computed embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the vector store from disk with deserialization allowed
    vector_store = FAISS.load_local(
        "embeddings", 
        embeddings,
        allow_dangerous_deserialization=True  # We trust our own embeddings
    )
    
    # Load metadata
    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {metadata['num_documents']} documents with {metadata['num_chunks']} chunks")
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create prompt template
    prompt = PromptTemplate(
        template=custom_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

# Initialize the chain at startup
conversation_chain = initialize_chain()

# Store conversation history
conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'answer': 'Please ask a question'})
    
    # Get response from the chain
    result = conversation_chain({"question": user_message, "chat_history": conversation_history})
    answer = result["answer"]
    
    # Update conversation history
    conversation_history.append((user_message, answer))
    
    return jsonify({'answer': answer})

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'success', 'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
