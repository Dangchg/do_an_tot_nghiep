import os
from typing import Optional, Literal
from dotenv import load_dotenv

# Import các thư viện lõi của LangChain
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load API Key
load_dotenv()

# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU (SCHEMA)
# ==========================================
class AdmissionQueryCheck(BaseModel):
    """
    Cấu trúc dữ liệu để kiểm tra tính đầy đủ của câu hỏi về ĐIỂM CHUẨN.
    """
    major: Optional[str] = Field(
        default=None, 
        description="Tên ngành hoặc Mã ngành (VD: 'IT1', 'Khoa học máy tính', 'Toán Tin')."
    )
    
    year: Optional[int] = Field(
        default=None, 
        description="Năm tuyển sinh (VD: 2023, 2024). Hãy convert các từ như 'năm ngoái', 'năm nay' thành số năm cụ thể nếu có thể."
    )
    
    method: Optional[Literal['THPT', 'TSA', 'XET_TUYEN', 'KHAC']] = Field(
        default=None, 
        description="Phương thức xét tuyển. Quy ước: 'THPT' (thi tốt nghiệp/đại học), 'TSA' (đánh giá tư duy/tư duy), 'XET_TUYEN' (học bạ/xét tuyển thẳng)."
    )
    
    missing_info_question: Optional[str] = Field(
        default=None, 
        description="Câu hỏi để hỏi lại người dùng nếu thiếu 1 trong 3 trường trên. Nếu đủ thì để None."
    )
    
    is_sufficient: bool = Field(
        description="True nếu ĐỦ cả 3 yếu tố (major, year, method). False nếu thiếu bất kỳ cái nào."
    )

# ==========================================
# 2. HÀM CHECK LOGIC
# ==========================================
def check_query_sufficiency(user_query: str) -> AdmissionQueryCheck:
    """
    Hàm phân tích câu hỏi người dùng để xem đã đủ thông tin chưa.
    
    Args:
        user_query (str): Câu hỏi của người dùng (hoặc ngữ cảnh đã gộp).
        
    Returns:
        AdmissionQueryCheck: Object chứa thông tin đã trích xuất và trạng thái đủ/thiếu.
    """
    
    # Sử dụng model nhỏ (mini) để tiết kiệm chi phí và tốc độ nhanh
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Ép kiểu đầu ra theo Schema đã định nghĩa ở trên
    structured_llm = llm.with_structured_output(AdmissionQueryCheck)

    # Viết Prompt hướng dẫn chi tiết
    system_prompt = """
    Bạn là một 'Validator' (Bộ kiểm duyệt) cho Chatbot Tuyển sinh Đại học Bách Khoa Hà Nội (HUST).
    
    Nhiệm vụ: Phân tích câu hỏi của người dùng khi họ muốn tra cứu ĐIỂM CHUẨN.
    
    Quy tắc nghiệp vụ:
    1. Để tra cứu chính xác, bắt buộc phải có đủ 3 thông tin:
       - **Ngành (Major)**: Tên hoặc mã.
       - **Năm (Year)**: Phải là con số cụ thể.
       - **Phương thức (Method)**: Phải xác định rõ là điểm thi THPT hay Đánh giá tư duy (TSA).
       
    2. Xử lý từ đồng nghĩa:
       - "Điểm thi đại học", "tốt nghiệp", "THPT" -> Method='THPT'.
       - "Đánh giá tư duy", "kỳ thi riêng", "TSA" -> Method='TSA'.
       
    3. Logic tạo câu hỏi (missing_info_question):
       - Chỉ hỏi lại phần thông tin bị thiếu.
       - Giọng điệu tự nhiên, lịch sự, ngắn gọn.
       
    Ví dụ Input -> Output:
    - Input: "Điểm chuẩn IT1" 
      -> Thiếu Year, Method -> Hỏi: "Bạn muốn xem điểm chuẩn IT1 của năm nào (VD: 2024) và theo phương thức nào (THPT hay Đánh giá tư duy)?"
      
    - Input: "Điểm TSA năm 2024" 
      -> Thiếu Major -> Hỏi: "Bạn muốn xem điểm Đánh giá tư duy năm 2024 của ngành nào?"
      
    - Input: "Điểm chuẩn IT1 2024 xét tư duy"
      -> Đủ -> missing_info_question = None, is_sufficient = True.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    # Tạo Chain và chạy
    chain = prompt | structured_llm
    result = chain.invoke({"query": user_query})
    
    return result

# ==========================================
# 3. PHẦN TEST ĐỘC LẬP (Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- TEST MODULE VALIDATOR ---\n")
    
    test_cases = [
        "Cho mình hỏi điểm chuẩn ngành Khoa học máy tính",         # Case 1: Thiếu Năm + Phương thức
        "Điểm chuẩn IT1 năm 2024",                                # Case 2: Thiếu Phương thức
        "Điểm đánh giá tư duy ngành Toán Tin năm 2024 là bao nhiêu", # Case 3: Đủ
        "Năm 2023"                                                # Case 4: Rời rạc (Chỉ có năm)
    ]
    
    for query in test_cases:
        print(f"🔹 Input: '{query}'")
        res = check_query_sufficiency(query)
        
        if res.is_sufficient:
            print(f"✅ ĐỦ THÔNG TIN -> Gọi SQL Agent")
            print(f"   Data: Major={res.major}, Year={res.year}, Method={res.method}")
        else:
            print(f"❌ THIẾU THÔNG TIN -> Dừng lại")
            print(f"   Bot hỏi lại: '{res.missing_info_question}'")
            print(f"   (Đã có: Major={res.major}, Year={res.year}, Method={res.method})")
        print("-" * 50)