import os
import glob
import time
import base64
from dotenv import load_dotenv
import PyPDF2 

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
import gradio as gr

# Các import Agent của bạn
from agent_tinh_cach import render_quiz_tab
from agnet_gioi_thieu_nganh import ask_advisor
from agnet_diem_chuan import ask_mysql
from agnet_quy_che import init_system2, chat_interface

# Import module dữ liệu
try:
    from seed_data import load_data_from_folder, vector_store2
except ImportError:
    print("⚠️  Chưa có seed_data.py. RAG sẽ không chạy được.")

# Load API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# ==============================================================================
# 🗂️ 1. MODULE ĐỌC TÀI LIỆU TẠM THỜI (PDF / ẢNH)
# ==============================================================================

def extract_text_from_pdf(pdf_path):
    """Trích xuất văn bản từ file PDF"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        return f"❌ Lỗi đọc PDF: {str(e)}"

def extract_text_from_image(image_path):
    """Sử dụng GPT-4o-mini Vision để đọc chữ từ ảnh"""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hãy trích xuất toàn bộ văn bản trong bức ảnh này. Giữ nguyên định dạng và bảng biểu nếu có."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = llm.invoke([msg])
        return response.content
    except Exception as e:
        return f"❌ Lỗi đọc Ảnh: {str(e)}"

def extract_text_from_file(filepath):
    """Hàm tổng hợp trích xuất text"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
        return extract_text_from_image(filepath)
    else:
        return f"❌ Định dạng '{ext}' chưa được hỗ trợ."

# ==============================================================================
# 🧠 2. ĐỊNH NGHĨA CÁC AGENT CON (WORKERS)
# ==============================================================================

def get_advisor_chain():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là chuyên gia hướng nghiệp. Hãy trả lời ngắn gọn, súc tích về các ngành học, cơ hội nghề nghiệp và thị trường lao động."),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# ==============================================================================
# 🚦 3. XÂY DỰNG ROUTER LLM (THE BRAIN)
# ==============================================================================

def get_router_chain():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) 
    system_message = """
    Bạn là một Router thông minh phân loại câu hỏi của học sinh.
    Hãy phân tích câu hỏi và trả về duy nhất MỘT từ khóa sau:
    1. "RAG": Quy chế, học phí, mã ngành...
    2. "ADVISOR": Định nghĩa ngành, cơ hội việc làm...
    3. "QUIZ": Trắc nghiệm...
    4. "TEXT2SQL": Điểm chuẩn...
    5. "CHAT": Xã giao...
    Chỉ trả về đúng từ khóa. Không giải thích thêm.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Câu hỏi: {question}")
    ])
    return prompt | llm | StrOutputParser()

router_chain = get_router_chain()
advisor_chain = get_advisor_chain()

# ==============================================================================
# 🎮 4. HÀM XỬ LÝ CHÍNH (ORCHESTRATOR)
# ==============================================================================

def main_chat_handler(message_dict, history):
    user_text = message_dict.get("text", "")
    files = message_dict.get("files", [])
    
    # ---------------------------------------------------------
    # TRƯỜNG HỢP 1: CÓ FILE ĐÍNH KÈM (PHÂN TÍCH TẠM THỜI)
    # ---------------------------------------------------------
    if files:
        combined_text = ""
        for file_path in files:
            filename = os.path.basename(file_path)
            extracted_text = extract_text_from_file(file_path)
            
            if "❌" in extracted_text or not extracted_text.strip():
                return f"⚠️ Xin lỗi, tôi không thể đọc được chữ từ file '{filename}'."
                
            combined_text += f"\n--- NỘI DUNG TÀI LIỆU {filename} ---\n{extracted_text}\n"

        # Nếu chỉ up file mà không hỏi gì
        if not user_text.strip():
            return "📄 Tôi đã đọc xong tài liệu. Bạn chưa đặt câu hỏi? Hãy gửi lại ảnh và đặt câu hỏi bạn mong muốn."
            
        # Trả lời dựa trên file vừa up
        llm_doc = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        doc_prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý đọc tài liệu. Hãy trả lời câu hỏi DỰA VÀO DUY NHẤT TÀI LIỆU ĐƯỢC CUNG CẤP. Nếu tài liệu không có thông tin, hãy nói: 'Tài liệu bạn cung cấp không chứa thông tin này'. Tuyệt đối không tự bịa ra hay lấy dữ liệu bên ngoài.\n\n{context}"),
            ("human", "{question}")
        ])
        
        doc_chain = doc_prompt | llm_doc | StrOutputParser()
        try:
            ans = doc_chain.invoke({"context": combined_text, "question": user_text})
            return f"👁️ **(Đọc file đính kèm)**:\n\n{ans}"
        except Exception as e:
            return f"Lỗi phân tích file: {e}"

    # ---------------------------------------------------------
    # TRƯỜNG HỢP 2: KHÔNG CÓ FILE ĐÍNH KÈM (DÙNG HỆ THỐNG RAG)
    # ---------------------------------------------------------
    if not user_text.strip():
        return "Bạn chưa nhập câu hỏi."
        
    try:
        route = router_chain.invoke({"question": user_text}).strip().upper()
        print(f"📡 Router Decision: [{route}]")
    except Exception as e:
        return f"Lỗi Router: {e}"

    if route == "RAG":
        return f"🤖 **(Tra cứu Dữ liệu Tuyển sinh)**:\n\n{init_system2()}{chat_interface(user_text, history)}"
    elif route == "ADVISOR":
        return f"🎓 **(Tư vấn Viên)**:\n{ask_advisor(user_text)}"
    elif route == "QUIZ":
        return "🧩 Hãy chuyển sang **Tab '🧩Trắc nghiệm Hướng nghiệp🧩'** ở phía trên màn hình nhé!"
    elif route == "TEXT2SQL":
        return f"🎓 **(Tư vấn Viên)**:\n {ask_mysql(user_text, history)}"
    else: 
        return "👋 Chào bạn! Tôi là trợ lý ảo tuyển sinh. Bạn có thể hỏi tôi về quy chế, điểm chuẩn hoặc định hướng ngành nghề."

# ==============================================================================
# 🖥️ 5. GIAO DIỆN (GRADIO)
# ==============================================================================
print("🚀 Đang khởi động hệ thống...")

with gr.Blocks(title="Hệ thống Tuyển sinh Đa phương thức", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 SUPER AGENT: TƯ VẤN TUYỂN SINH & HƯỚNG NGHIỆP")
    
    with gr.Tabs():
        with gr.TabItem("💬 Trợ lý ảo AI"):
            custom_chatbot_component = gr.Chatbot(
                height=600,             
                placeholder="👋 Xin chào! Hãy đặt câu hỏi hoặc tải ảnh/tài liệu lên bằng nút '+' bên dưới."
            )

            gr.ChatInterface(
                fn=main_chat_handler,
                multimodal=True,
                chatbot=custom_chatbot_component,
                examples=[
                    {"text": "Tôi sẽ hợp với nghề gì?"}, 
                    {"text": "Điểm chuẩn ngành Toán Tin Bách Khoa Hà Nội?"}
                ]
            )
        
        with gr.TabItem("🧩 Trắc nghiệm Hướng nghiệp🧩"):
            render_quiz_tab()

if __name__ == "__main__":
    demo.launch(share=False)