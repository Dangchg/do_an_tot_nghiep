import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import gradio as gr



from pydantic import BaseModel, Field
from typing import Optional, List

class QueryIntent(BaseModel):
    """Lưu trữ ý định người dùng và metadata trích xuất được."""
    
    truong: Optional[List[str]] = Field(
        default=None, 
        description="Mã các trường đại học được nhắc đến. Cần map từ tên gọi sang mã: Bách Khoa Hà nội -> hust, Kinh tế Quốc dân -> neu, Ngoại thương -> ftu, Bách khoa HCM -> hcmut..."
    )
    nam: Optional[List[str]] = Field(
        default=None, 
        description="Năm tuyển sinh (ví dụ: '2023', '2024', '2025')"
    )
    doc_type: Optional[str] = Field(
        default=None, 
        description="Loại tài liệu:  'de_an_tuyen_sinh', 'thong_bao'."
    )
    
    status: str = Field(
        description="Trạng thái phân tích. Trả về 'ready' nếu đã đủ dữ kiện tìm kiếm cơ bản (ÍT NHẤT PHẢI CÓ TÊN TRƯỜNG). Trả về 'need_more_info' nếu câu hỏi quá chung chung và KHÔNG CÓ TÊN TRƯỜNG."
    )
    ask_user: Optional[str] = Field(
        default=None, 
        description="Nếu status là 'need_more_info', hãy viết một câu hỏi lịch sự ngắn gọn để hỏi thêm người dùng thông tin còn thiếu (ví dụ: tên trường, năm)."
    )



from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def extract_filter_from_query(user_query: str, history_text: str = "") -> QueryIntent:
    """
    Phân tích câu hỏi và lịch sử để trích xuất metadata filter.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(QueryIntent)
    
    system_prompt = """
    Bạn là chuyên gia phân tích yêu cầu tuyển sinh.
    Nhiệm vụ của bạn là trích xuất các điều kiện lọc (metadata filter).
    
    Quy tắc bắt buộc:
    1. QUAN TRỌNG NHẤT: Phải biết người dùng đang hỏi về TRƯỜNG NÀO. 
    2. KẾ THỪA NGỮ CẢNH: Nếu người dùng KHÔNG nhắc đến tên trường trong CÂU HỎI HIỆN TẠI, bạn PHẢI ĐỌC LỊCH SỬ. Nếu lượt chat ngay trước đó đang nói về trường nào, hãy tự động lấy mã trường đó điền vào kết quả.
    3. Chỉ khi CẢ lịch sử và câu hỏi hiện tại đều không có thông tin trường, mới set status = 'need_more_info' và hỏi lại người dùng.
    4. Chuyển đổi tên trường sang mã viết tắt:
       - Đại học Bách Khoa Hà Nội, Bách Khoa, HUST -> hust
       - Đại học Kinh tế Quốc dân, NEU -> neu
       - Đại học Ngoại thương, FTU -> ftu
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "LỊCH SỬ HỘI THOẠI:\n{history}\n\nCÂU HỎI HIỆN TẠI:\n{query}")
    ])
    
    chain = prompt | structured_llm
    
    # Chạy chuỗi với cả lịch sử và câu hỏi
    result = chain.invoke({
        "query": user_query,
        "history": history_text
    })
    return result



# Import pipeline mới (seed_datacopy.py)
from seed_data import (
    load_data_from_folder,
    build_vectorstore,
    load_vectorstore,
    build_chroma_filter,
)

load_dotenv()
os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ─────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────
global_chain   = None
global_chunks  = None   # giữ lại chunks cho BM25 (cần nội dung text)
global_vectorstore = None

# ─────────────────────────────────────────────────────────────────────
# 1. HYBRID RETRIEVER — hỗ trợ metadata filter cho vector retriever
# ─────────────────────────────────────────────────────────────────────

def create_hybrid_retriever(vectorstore, chunks, filter_meta: dict = None):
    """
    Tạo bộ tìm kiếm lai BM25 + Chroma Semantic.
    filter_meta: lọc theo metadata khi query vector store.
      Ví dụ: {"truong": "hust"} hoặc {"truong": "hust", "nam": "2025"}
    """
    print("🔍 Đang cấu hình Hybrid Retrieval...")

    # --- BM25 (keyword) ---
    # Lọc chunks theo metadata trước khi đưa vào BM25 (nếu có filter)
    if filter_meta:
        filtered_chunks = [
            c for c in chunks
            if all(c.metadata.get(k) == v for k, v in filter_meta.items())
        ]
        print(f"   BM25 dùng {len(filtered_chunks)}/{len(chunks)} chunks sau filter")
    else:
        filtered_chunks = chunks

    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = 10

    # --- Chroma (semantic) với metadata filter ---
    chroma_filter   = build_chroma_filter(filter_meta)
    search_kwargs   = {"k": 10}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    chroma_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    # --- Ensemble ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble_retriever


# ─────────────────────────────────────────────────────────────────────
# 2. RAG CHAIN
# ─────────────────────────────────────────────────────────────────────

def setup_rag_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True,
    )

    system_template = (
        "Bạn là một chuyên gia tư vấn tuyển sinh đại học tại Việt Nam. "
        "Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến điểm tuyển sinh đầu vào, "
        "quy chế tuyển sinh, trường học, các phương thức tuyển sinh và cách tính điểm "
        "theo từng phương thức một cách ngắn gọn và chính xác. "
        "Nếu bạn không biết câu trả lời, hãy nói rõ rằng bạn không biết. "
        "Tuyệt đối không bịa ra thông tin nếu không có ngữ cảnh liên quan được cung cấp."
    )

    human_template = """Sử dụng thông tin ngữ cảnh sau để trả lời câu hỏi:

Ngữ cảnh (Context):
{context}

Câu hỏi (Question): {question}
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
    )
    return qa_chain


# ─────────────────────────────────────────────────────────────────────
# 3. INIT — tạo mới hoặc load DB có sẵn
# ─────────────────────────────────────────────────────────────────────

def init_system(rebuild: bool = False, filter_meta: dict = None):
    """
    rebuild=True  → đọc file, tạo lại vector DB
    rebuild=False → load vector DB đã lưu, vẫn cần đọc file cho BM25
    filter_meta   → giới hạn theo trường/năm/loại tài liệu
    """
    global global_chain, global_chunks, global_vectorstore

    # Luôn load chunks cho BM25 (không cần LLM enrichment khi chỉ load)
    print("📂 Đang load documents...")
    global_chunks = load_data_from_folder(
        root="Data",
        use_llm_enrichment=rebuild,   # chỉ enrich khi rebuild
        llm_sample_limit=200,
    )

    if rebuild:
        print("🔄 Tạo lại Vector DB...")
        vectorstore = build_vectorstore(global_chunks, db_name="vector_db", device="cuda", rebuild=True)
    else:
        print("📂 Load Vector DB đã có...")
        vectorstore = load_vectorstore(db_name="vector_db", device="cpu")
    global_vectorstore = vectorstore
    hybrid_retriever = create_hybrid_retriever(vectorstore, global_chunks, filter_meta=filter_meta)
    global_chain = setup_rag_chain(hybrid_retriever)
    print("✅ Hệ thống sẵn sàng!")



# ─────────────────────────────────────────────────────────────────────
# 4. GRADIO CHAT
# ─────────────────────────────────────────────────────────────────────

def chat_interface(message, history):
    if global_vectorstore is None or global_chunks is None:
        return "⚠️ Hệ thống đang khởi động, vui lòng đợi..."

    # --- CHUYỂN BƯỚC ĐỌC LỊCH SỬ LÊN ĐẦU ---
    # Chuyển lịch sử chat thành dạng text để LLM dễ đọc
    history_text = ""
    for turn in history[-5:]:
        # Lưu ý: Gradio truyền history format [{"role": "user", "content": "..."}, ...]
        role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        history_text += f"{role}: {turn['content']}\n"

    # 1. Truyền cả message và history_text vào bộ trích xuất
    print("\n🧠 Đang phân tích ý định người dùng (kèm lịch sử)...")
    intent = extract_filter_from_query(message, history_text)
    
    # 2. Xử lý trường hợp hỏi trống không (không có tên trường cả ở hiện tại lẫn lịch sử)
    if intent.status == "need_more_info":
        return intent.ask_user

    # 3. Tạo bộ lọc CỨNG
    dynamic_filter = {}
    if intent.truong:
        truong_code = intent.truong[0] if isinstance(intent.truong, list) else intent.truong
        dynamic_filter["truong"] = truong_code
        
    if intent.nam:
        nam_code = intent.nam[0] if isinstance(intent.nam, list) else intent.nam
        dynamic_filter["nam"] = nam_code

    print(f"🔒 Đang khóa không gian tìm kiếm với bộ lọc: {dynamic_filter}")

    # --- KIỂM TRA DỮ LIỆU ---
    if dynamic_filter:
        test_chunks = [
            c for c in global_chunks
            if all(c.metadata.get(k) == v for k, v in dynamic_filter.items())
        ]
        if len(test_chunks) == 0:
            return f"Xin lỗi, hiện tại tôi chưa có dữ liệu tuyển sinh của trường này (hoặc năm {dynamic_filter.get('nam', '')} mà bạn yêu cầu)."

    # 4. TẠO RETRIEVER
    strict_retriever = create_hybrid_retriever(
        global_vectorstore, 
        global_chunks, 
        filter_meta=dynamic_filter
    )
    
    # 5. Bọc lại vào chain
    strict_chain = setup_rag_chain(strict_retriever)

    # 6. Truy vấn (Vẫn giữ history text kèm theo để LLM trả lời mượt)
    try:
        response = strict_chain.invoke({
            "question": f"{history_text}Người dùng: {message}"
        })

        if isinstance(response, dict):
            return response.get("answer") or response.get("output") or response.get("result") or str(response)
        return str(response)

    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


# ─────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 HỆ THỐNG TRỢ LÝ TUYỂN SINH — Hybrid RAG")
    print("=" * 50)
    print("1. Tạo lại Vector DB từ đầu (có LLM metadata enrichment)")
    print("2. Load Vector DB đã có (nhanh hơn)")
    print("=" * 50)

    choice = input("👉 Nhập lựa chọn (1 hoặc 2): ").strip()

    # --- Tuỳ chọn filter theo trường / năm ---
    # Bỏ comment và chỉnh sửa nếu muốn giới hạn phạm vi tìm kiếm:
    # FILTER = {"truong": "hust"}
    # FILTER = {"truong": "hust", "nam": "2025"}
    FILTER = None   # None = không filter, tìm toàn bộ DB

    if choice == "1":
        init_system(rebuild=True,  filter_meta=FILTER)
    else:
        init_system(rebuild=False, filter_meta=FILTER)

    print("🌐 Đang mở giao diện Gradio...")
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        title="🎓 Trợ lý Tuyển sinh Đại học (Hybrid RAG)",
        description=(
            "Hỏi đáp thông tin tuyển sinh sử dụng tìm kiếm lai (Từ khóa + Ngữ nghĩa).\n"
            f"Bộ lọc hiện tại: {'Toàn bộ dữ liệu' if FILTER is None else str(FILTER)}"
        ),
        examples=[
            "Xét tuyển tài năng 2026 bưu chính viễn thông",
            "Các phương thức xét tuyển của Đại học Bách Khoa Hà Nội?",
            "Ngành Trí tuệ nhân tạo xét tuyển những tổ hợp môn nào?",
        ],
    ).launch()
