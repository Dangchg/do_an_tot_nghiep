import os
import glob
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import gradio as gr
from openai import OpenAI
import functools
from concurrent.futures import ThreadPoolExecutor
from g4f.client import Client
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from typing import List, Dict
from seed_data import load_data_from_folder, vector_store
client = Client()

# Performance improvement: Use caching for context loading
@functools.lru_cache(maxsize=None)
def load_context():
    """Load and cache context data to avoid repeated file I/O"""
    context = {}

    #Load tung truong voi luong cho nhanh 
    def load_data_files():
        data_context = {}
        data = glob.glob("Data/de_an_tuyen_sinh/*")

        def load_single_data (data):
            # 1. Lấy tên file (ví dụ: "file ve hoc.md")
            filename_with_ext = os.path.basename(data)
            
            # 2. Tách tên file và phần mở rộng (ví dụ: "file ve hoc" và ".md")
            filename_no_ext, _ = os.path.splitext(filename_with_ext)
            
            # 3. Áp dụng logic lấy từ cuối của bạn
            name = filename_no_ext.split(' ')[-1] # Kết quả sẽ là "hoc" (không có dấu chấm)
            try:
                with open(data, "r", encoding="utf-8") as f:
                    return name, f.read()
            except Exception as e:
                print(f"Loi tai len {data}: {e}")
                return name, ""
        
        # Use ThreadPoolExecutor for concurrent file loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(load_single_data, data)
            data_context.update(dict(results))
        
        return data_context
    
    # Load both types concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        data_future = executor.submit(load_data_files)
        
        context.update(data_future.result())
    return context

system_message = (
    "Bạn là một chuyên gia tư vấn tuyển sinh đại học tại Việt Nam . "
    "Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến điểm tuyển sinh đầu vào, quy chế tuyển sinh, trường học, các phương thức tuyển sinh và cách tính điểm theo cách phương thức tuyển sinh một cách ngắn gọn và chính xác. "
    "Nếu bạn không biết câu trả lời, hãy nói rõ rằng bạn không biết. "
    "Tuyệt đối không bịa ra thông tin nếu không có ngữ cảnh liên quan được cung cấp."
)

def get_relevant_context(message):
    context = load_context()
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context  

def add_context(message):
    """Add relevant context to message"""
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nNhững thông tin sau có thể hữu ích cho việc trả lời câu hỏi này:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message

def chat(message, history):
    """Optimized chat function with better error handling"""
    try:
        messages = [{"role": "system", "content": system_message}] + history
        message = add_context(message)
        messages.append({"role": "user", "content": message})

        stream = client.chat.completions.create(
            model="", 
            messages=messages, 
            stream=True,
            max_tokens=1000,  # Limit response length for faster generation
            temperature=0.7
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response
    except Exception as e:
        yield f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

def chatbox():
    # 2. Gọi hàm Load Data
    chunks = load_data_from_folder()

    if not chunks:
        print("Dừng chương trình do không có dữ liệu.")
        return
    
    # 3. Gọi hàm tạo Vector Store
    # Lưu ý: Nhận về vectorstore (để query) và fig (để vẽ)

    fig, vectorstore = vector_store(chunks)

    # 4. Hiển thị biểu đồ (nếu có)
    if fig:
        print("\nĐang mở biểu đồ trên trình duyệt...")
        fig.show()

    # Tạo mô hình Chat với OpenAI
    llm = ChatOpenAI(
        api_key="sk-bnPOclUlNLW7xF40Xdi35PtWhXA2k8S6gprHkeHGP9XuWQY7",
        base_url="https://gpt1.shupremium.com/v1",
        temperature=0.7, 
        model_name="gpt-4o-mini",
    )

    # Thiết lập bộ nhớ hội thoại
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True)


    # Tạo retriever từ vector store (Chroma)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6, "k": 10}
    )


    # 1. Fix keyword retriever to search content, not just title
    class KeywordRetriever(BaseRetriever):
        context_dict: Dict[str, str]

        def _get_relevant_documents(self, query: str) -> List[Document]:
            relevant_docs = []
            for title, content in self.context_dict.items():
                if any(kw in content.lower() for kw in query.lower().split()):
                    relevant_docs.append(Document(page_content=content, metadata={"source": title}))
            return relevant_docs

        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return self.get_relevant_documents(query)

    # 2. Use keyword + properly configured vector retriever
    context = load_context()
    keyword_retriever = KeywordRetriever(context_dict=context)
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 100}
    )

    # 3. Ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    # 4. Conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        callbacks=[StdOutCallbackHandler()]
    )


    return conversation_chain

def view():
    def chat(question, history):
        conversation_chain = chatbox()
        result = conversation_chain.invoke({"question": question})
        return result["answer"]

    view = gr.ChatInterface(chat, type="messages").launch()
    return view

