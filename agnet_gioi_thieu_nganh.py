import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# Load environment variables
load_dotenv()

# Khởi tạo Chain (Singleton)
_advisor_chain = None

# ---------------------------------------------------------
# HÀM TẠO AGENT GIỚI THIỆU NGÀNH (DIRECT LLM)
# ---------------------------------------------------------

def get_advisor_chain():
    """Lazy load chain để tránh khởi tạo thừa"""
    global _advisor_chain
    if _advisor_chain is not None:
        return _advisor_chain
    
    # Setup LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # 2. Định nghĩa Prompt
    system_message = """
    Bạn là một Chuyên gia Tư vấn Hướng nghiệp và Phân tích Thị trường Lao động hàng đầu tại Việt Nam.
    
    Khi người dùng nhập tên một ngành học hoặc lĩnh vực quan tâm, hãy phân tích sâu sắc theo cấu trúc sau:
    
    1. 📖 **Tổng quan:** Ngành này là gì? (Giải thích đơn giản, dễ hình dung).
    2. 🎓 **Chương trình học:** Các môn học cốt lõi và kỹ năng sẽ được rèn luyện.
    3. 💼 **Vị trí việc làm:** Cụ thể các chức danh (Job titles) sau khi ra trường.
    4. 💰 **Thị trường & Thu nhập:** Mức lương khởi điểm, lương lâu năm và độ "khát" nhân lực tại Việt Nam hiện nay.
    5. 🎯 **Các công ty lớn trong ngành và vị trí việc làm thực tế :** Liệt kê các vị trí tuyển dụng của các công ty lớn trên.
    
    Hãy trả lời bằng định dạng Markdown đẹp mắt, sử dụng icon đầu dòng.
    Nếu người dùng nhập thứ không phải ngành học (ví dụ: "xin chào"), hãy trả lời xã giao ngắn gọn và hỏi họ quan tâm ngành nào.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])
    
    _advisor_chain = prompt | llm | StrOutputParser()
    return _advisor_chain



# ---------------------------------------------------------
# MAIN & GIAO DIỆN
# ---------------------------------------------------------

'''# Khởi tạo Chain toàn cục
major_advisor_chain = setup_major_advisor_chain()

def consult_major(message, history):
    """Hàm xử lý cho Gradio"""
    if not message.strip():
        return "Bạn hãy nhập tên ngành muốn tìm hiểu nhé!"
    
    try:
        # Gọi trực tiếp LLM
        response = major_advisor_chain.invoke({"major_name": message})
        return response
    except Exception as e:
        return f"Lỗi kết nối: {str(e)}"

# Chạy thử
if __name__ == "__main__":
    print("🚀 Đang khởi động Agent Tư vấn Ngành học...")
    
    examples = [
        ["Khoa học máy tính"],
        ["Logistics và Quản lý chuỗi cung ứng"],
        ["Marketing số (Digital Marketing)"],
        ["Công nghệ kỹ thuật ô tô"]
    ]

    gr.ChatInterface(
        consult_major,
        type="messages",
        title="🎓 Chuyên gia Hướng nghiệp AI",
        description="Nhập tên ngành học bạn quan tâm (Ví dụ: CNTT, Y Đa khoa...), AI sẽ phân tích chi tiết cho bạn.",
        examples=examples
    ).launch()'''

def ask_advisor(message):
    """Hàm giao tiếp chính được gọi từ Router"""
    try:
        chain = get_advisor_chain()
        return chain.invoke({"question": message})
    except Exception as e:
        return f"Lỗi Advisor: {str(e)}"