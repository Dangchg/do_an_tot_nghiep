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

    # 3. Custom System Prompt cho bối cảnh HUST
    system_prompt = """
    Quy trình suy luận (BẮT BUỘC TUÂN THỦ):
    
    BƯỚC 1: KIỂM TRA THÔNG TIN ĐẦU VÀO
    Trước khi tạo bất kỳ câu lệnh SQL nào, hãy kiểm tra xem người dùng đã cung cấp đủ 3 yếu tố sau chưa:
    1. **Ngành học:** (VD: IT1, Toán Tin, hoặc tên ngành chung).
    2. **Năm tuyển sinh:** (VD: 2023, 2024). Nếu thiếu, KHÔNG ĐƯỢC tự ý chọn năm hiện tại.
    3. **Phương thức xét tuyển:** (VD: Thi THPT, Đánh giá tư duy/TSA).
    
    BƯỚC 2: QUYẾT ĐỊNH HÀNH ĐỘNG
    - **TRƯỜNG HỢP THIẾU THÔNG TIN:** Nếu thiếu Năm hoặc Phương thức: Hãy dừng lại và hỏi người dùng.
      -> Ví dụ: "Bạn muốn xem điểm chuẩn năm nào và theo phương thức xét tuyển nào (THPT hay Đánh giá tư duy)?"
      -> Ví dụ: "Ngành CNTT có ở cả Bách Khoa và Đại học Công Nghệ, bạn muốn hỏi trường nào?"
      
    - **TRƯỜNG HỢP ĐỦ THÔNG TIN:**
      Tiến hành tạo SQL query.
      Lưu ý: Luôn `SELECT` cột `year`, `major_code`, `method_name` để hiển thị rõ ràng.
      
    BƯỚC 3: TRẢ LỜI
    - Nếu kết quả SQL rỗng: Trả lời "Không tìm thấy dữ liệu cho tiêu chí bạn chọn".
    - Nếu đủ điều kiện trả lời điểm của đúng ngành đó không cần trả về điểm chuẩn của các ngành khác. Trả lời thật đúng trọng tâm không cần thêm bớt.
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

def ask_mysql(message, history):
    try:
        agent = get_mysql_agent()
        response = agent.invoke({"input": message})
        return response["output"]
    except Exception as e:
        return f"⚠️ Lỗi hệ thống: {str(e)}"

# --- TEST ---
if __name__ == "__main__":
    # Test 1: Hỏi mã ngành (Đặc sản HUST)
    q1 = "Điểm chuẩn ngành MI1 năm 2024 xét bằng điểm thi tốt nghiệp là bao nhiêu?"
    print(f"\nUser: {q1}")
    print(f"Bot: {ask_mysql(q1)}")

    # Test 2: Hỏi TSA
    q2 = "Điểm đánh giá tư duy ngành Khoa học máy tính Bách Khoa?"
    print(f"\nUser: {q2}")
    print(f"Bot: {ask_mysql(q2)}")