import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_classic.memory import ConversationBufferWindowMemory

# Load biến môi trường
load_dotenv()

# Cấu hình kết nối MySQL
# Bạn nên lưu các thông tin này trong file .env để bảo mật
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "dangnxh2004")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "do_an")
DB_PORT = "3306"

# Chuỗi kết nối SQLAlchemy cho MySQL
# Format: mysql+mysqlconnector://user:pass@host:port/db_name
MYSQL_URI = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

_sql_agent_executor = None

def get_mysql_agent():
    global _sql_agent_executor
    if _sql_agent_executor is not None:
        return _sql_agent_executor

    print(f"🔌 Đang kết nối đến MySQL: {DB_NAME}...")
    
    # 1. Khởi tạo SQLDatabase từ URI
    # include_tables: Chỉ cho phép Agent nhìn thấy các bảng này (Security)
    db = SQLDatabase.from_uri(
        MYSQL_URI,
        include_tables=['universities', 'majors', 'admission_scores'],
        sample_rows_in_table_info=3 # Lấy 3 dòng mẫu để LLM hiểu dữ liệu
    )

    # 2. Setup LLM (Dùng model thông minh để viết SQL chuẩn)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Bộ nhớ hội thoại (Nhớ 5 cặp câu hỏi-trả lời gần nhất)
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True
    )

    # 3. Custom System Prompt cho bối cảnh 
    system_prompt = """
    Bạn là Tác nhân AI truy vấn cơ sở dữ liệu tuyển sinh đại học (Text2SQL Agent).
    Quy trình suy luận của bạn BẮT BUỘC TUÂN THỦ NGHIÊM NGẶT các bước sau:
    
    BƯỚC 1: KIỂM TRA THÔNG TIN ĐẦU VÀO
    Đánh giá xem câu hỏi của người dùng đã có đủ 3 yếu tố cốt lõi chưa:
    1. Ngành học: (Mã ngành hoặc tên ngành).
    2. Năm tuyển sinh: Nếu người dùng không nói rõ năm, KHÔNG ĐƯỢC tự ý chọn năm hiện tại.
    3. Phương thức xét tuyển: (VD: THPT, Đánh giá tư duy/TSA, Học bạ).
    
    BƯỚC 2: QUYẾT ĐỊNH HÀNH ĐỘNG
    - Nếu THIẾU Năm hoặc Phương thức xét tuyển: Dừng ngay việc tạo SQL và hỏi lại người dùng. 
      VD: "Bạn muốn xem điểm chuẩn năm nào và theo phương thức xét tuyển nào (THPT hay Đánh giá tư duy)?"
    - Nếu ĐỦ thông tin: Tiến hành tạo truy vấn SQL theo các quy tắc ở Bước 3.
    
    BƯỚC 3: CHIẾN LƯỢC TÌM KIẾM CHUỖI MỜ (QUY TẮC SỐNG CÒN)
    Cơ sở dữ liệu rất nhạy cảm với khoảng trắng, dấu phẩy và chính tả. TUYỆT ĐỐI KHÔNG dùng toán tử `=` khi truy vấn chuỗi văn bản (tên ngành, phương thức). BẮT BUỘC dùng `LIKE` kết hợp các kỹ thuật sau:
    
    1. Cắt nhỏ từ khóa: Bỏ các từ đệm, thay khoảng trắng bằng `%`. 
       VD: "Công nghệ Dệt May" -> `WHERE major_name LIKE '%Dệt%May%'`.
       
    2. Xử lý chính tả tiếng Việt (i / y): Nếu từ khóa có âm 'i' hoặc 'y', bắt buộc thay ký tự đó bằng dấu `_` (đại diện 1 ký tự).
       VD: "Kĩ thuật" hoặc "Kỹ thuật" -> `LIKE '%K_ thuật%'`.
       VD: "Vật lí" hoặc "Vật lý" -> `LIKE '%Vật l_%'`.
       
    3. Đồng nghĩa Phương thức xét tuyển (BẮT BUỘC DÙNG OR):
       - Nhóm Đánh giá tư duy: Nếu truy vấn nhắc đến "TSA", "đánh giá tư duy", "ĐGTD" -> `(method_name LIKE '%đánh giá tư duy%' OR method_name LIKE '%TSA%' OR method_name LIKE '%ĐGTD%')`
       - Nhóm Đánh giá năng lực: Nếu nhắc đến "HSA", "APT", "đánh giá năng lực", "ĐGNL" -> `(method_name LIKE '%đánh giá năng lực%' OR method_name LIKE '%HSA%' OR method_name LIKE '%ĐGNL%')`
       - Nhóm THPT: Nếu nhắc đến "THPT", "tốt nghiệp", "đại trà" -> `(method_name LIKE '%THPT%' OR method_name LIKE '%tốt nghiệp%')`
       - Nhóm Học bạ: Nếu nhắc đến "học bạ", "kết quả học tập" -> `(method_name LIKE '%học bạ%' OR method_name LIKE '%học tập%')`
       
    4. Kỹ thuật truy vấn chéo (Khuyến nghị): Nên thực hiện một câu lệnh `SELECT major_code FROM majors WHERE...` bằng kỹ thuật `LIKE` trước để lấy chính xác mã ngành, sau đó mới dùng mã đó để truy vấn điểm trong bảng `admission_scores`.

    BƯỚC 4: ĐỊNH DẠNG KẾT QUẢ ĐẦU RA
    - Câu lệnh SQL luôn phải `SELECT` các cột cốt lõi: `year`, `major_code`, `major_name`, `method_name`, `score` để có ngữ cảnh đầy đủ.
    - Nếu SQL không trả về kết quả nào (Empty set): Trả lời chính xác câu "Hiện tại tôi chưa có dữ liệu điểm chuẩn cho tiêu chí bạn chọn."
    - Nếu có kết quả: Chỉ trả lời ĐÚNG TRỌNG TÂM mức điểm của ngành và phương thức được hỏi. Tuyệt đối không liệt kê lan man các ngành khác không liên quan.
    5. Xử lý tên ngành ghép (chứa chữ "và", dấu "-"):
       Nhiều ngành học có tên ghép rất dài chứa liên từ "và" (VD: "Tiếng Anh KHKT và Công nghệ", "Kỹ thuật Cơ điện tử và Robot").
       - TUYỆT ĐỐI KHÔNG chia tách truy vấn thành 2 ngành riêng biệt nếu chúng nằm liền nhau trong ngữ cảnh một trường.
       - Cách xử lý: Hãy coi toàn bộ cụm đó là một ngành duy nhất, xóa bỏ chữ "và" hoặc dấu "-" và thay thế bằng ký tự `%`.
       - VD: Người dùng hỏi "Tiếng Anh KHKT và Công nghệ" -> BẮT BUỘC query: `WHERE major_name LIKE '%Tiếng Anh%KHKT%Công nghệ%'`.
    """

    # 4. Tạo Agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    _sql_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True,
        memory=memory,
        suffix=system_prompt
    )
    
    return _sql_agent_executor

def ask_mysql(history_text, user_text):
    try:
        agent = get_mysql_agent()

        response = agent.invoke({
            "input": f"""
Lịch sử:
{history_text}

Câu hỏi:
{user_text}
"""
        })

        if isinstance(response, dict):
            return (
                response.get("output")
                or response.get("answer")
                or response.get("result")
                or str(response)
            )
        else:
            return str(response)

    except Exception as e:
        return f"⚠️ Lỗi hệ thống: {str(e)}"

# --- TEST ---
if __name__ == "__main__":
    # Test 1: Hỏi mã ngành (Đặc sản HUST)
    q1 = "Điểm chuẩn ngành MI1 năm 2024 xét bằng điểm thi tốt nghiệp là bao nhiêu?"
    print(f"\nUser: {q1}")
    print(f"Bot: {ask_mysql(q1)}")

    # Test 2: Hỏi TSA
    q2 = "Điểm chuẩn kinh tế quốc dân?"
    print(f"\nUser: {q2}")
    print(f"Bot: {ask_mysql(q2)}")
