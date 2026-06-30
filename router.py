import os
import glob
import time
import atexit 
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed

# Các import Agent của bạn
from agent_tinh_cach import render_quiz_tab
from agent_gioi_thieu_nganh import ask_advisor
from agent_diem_chuan import ask_mysql

# Import RAG Agent (Đã sửa lại để gọi init 1 lần)
from agent_quy_che import init_system, chat_interface

# [THÊM MỚI] IMPORT DATABASE MANAGER
try:
    from db_manager import save_message_and_keep_top_5, get_user_history
except ImportError:
    print("⚠️ Không tìm thấy file 'db_manager.py'. Vui lòng tạo file này để quản lý MySQL.")


try:
    from auth_ui import create_auth_components, bind_auth_events
except ImportError:
    print("⚠️ Không tìm thấy file 'auth_ui.py'. Tính năng đăng nhập sẽ bị lỗi.")

try:
    from ocr_agent import extract_text_via_ocr_api, stop_ocr_server
    atexit.register(stop_ocr_server)
except ImportError:
    print("⚠️ Không tìm thấy file 'ocr_agent.py'. Tính năng đọc tài liệu sẽ bị lỗi.")
    def extract_text_via_ocr_api(path): return "❌ Thiếu module ocr_agent."

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")


# ==============================================================================
# 🧠 1. XỬ LÝ LỊCH SỬ (HISTORY FORMATTERS)
# ==============================================================================

def format_history_for_router(history_rows):
    """Định dạng lịch sử thành chuỗi văn bản cho Router phân tích ý định."""
    formatted = []
    # Chỉ lấy 4 lượt gần nhất để tránh Router bị nhầm lẫn bởi ngữ cảnh quá xa
    for row in history_rows[-4:]:
        role = "User" if row["role"] == "human" else "AI"
        formatted.append(f"{role}: {row['content']}")
    return "\n".join(formatted).strip()

def format_history_for_rag(history_rows):
    """Định dạng lịch sử thành List[Dict] đúng chuẩn Gradio cho agent_quy_che."""
    gradio_hist = []
    for row in history_rows:
        role = "user" if row["role"] == "human" else "assistant"
        gradio_hist.append({"role": role, "content": row["content"]})
    return gradio_hist


# ==============================================================================
# 🗂️ 2. MODULE ĐỌC TÀI LIỆU 
# ==============================================================================

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file: return file.read()
    except Exception:
        try:
            with open(txt_path, "r", encoding="windows-1258") as file: return file.read()
        except Exception as e2:
            return f"❌ Lỗi đọc file TXT: {str(e2)}"
        
def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        return extract_text_from_txt(filepath)
    elif ext in ['.pdf', '.jpg', '.jpeg', '.png', '.webp']:
        return extract_text_via_ocr_api(filepath)
    else:
        return f"❌ Định dạng '{ext}' chưa được hỗ trợ."


# ==============================================================================
# 🚦 3. XÂY DỰNG ROUTER LLM (THE BRAIN)
# ==============================================================================

def get_router_chain_with_memory():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    system_message = """
    Bạn là Router AI thông minh. Phân loại câu hỏi của học sinh.
    1. RAG: Quy chế, học phí, mã ngành, chỉ tiêu tuyển sinh 
    2. ADVISOR: Định nghĩa ngành, nghề nghiệp, môn học
    3. QUIZ: Trắc nghiệm hướng nghiệp
    4. TEXT2SQL: Câu hỏi về điểm chuẩn
    5. CHAT: Hội thoại bình thường, chào hỏi, cảm ơn.

    QUAN TRỌNG:
    - Nếu câu hỏi hỏi nhiều ý -> Trả về nhiều nhãn cách nhau bằng dấu phẩy (vd: ADVISOR,TEXT2SQL).
    - Tối đa 2 nhãn.
    - Chỉ trả về nhãn, KHÔNG giải thích thêm.
    """
    
    examples = [
        {"question": "Học phí ngành Y năm 2024 là bao nhiêu?", "history": "", "label": "RAG"},
        {"question": "Ngành Data Science học những môn gì?", "history": "", "label": "ADVISOR"},
        {"question": "Điểm chuẩn ĐH Bách Khoa 2023 ngành CNTT?", "history": "", "label": "TEXT2SQL"},
        {"question": "Tôi không biết chọn ngành gì", "history": "", "label": "QUIZ"},
        {"question": "Cảm ơn bạn nhiều nhé!", "history": "", "label": "CHAT"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "LỊCH SỬ: {history}\nCÂU HỎI: {question}"),
        ("ai", "{label}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        few_shot_prompt,
        ("human", "Lịch sử hội thoại:\n{history}\n\nCâu hỏi hiện tại:\n{question}\n\nXác định nhãn phù hợp (VD: RAG hoặc ADVISOR,TEXT2SQL).")
    ])

    return prompt | llm | StrOutputParser()

router_chain = get_router_chain_with_memory()


# ==============================================================================
# 🎮 4. HÀM XỬ LÝ CHÍNH (ORCHESTRATOR)
# ==============================================================================

# Khởi tạo VectorDB 1 lần duy nhất khi bật file!
print("⏳ Khởi tạo RAG System (VectorDB)...")
try:

    init_system(rebuild=False, filter_meta=None) 
except Exception as e:
    print(f"⚠️ Không thể khởi tạo RAG (agent_quy_che): {e}")

# Chú ý chữ ký hàm (signature): lambda h_text, q, h_raw
AGENT_MAP = {
    "RAG": (
        "🏫 **Quy chế & Tuyển sinh**",
        lambda h_text, q, h_raw: chat_interface(q, h_raw)  # Truyền h_raw (list) cho RAG
    ),
    "ADVISOR": (
        "🎓 **Tư vấn ngành**",
        lambda h_text, q, h_raw: ask_advisor(h_text, q)    # Truyền h_text cho Advisor
    ),
    "TEXT2SQL": (
        "📊 **Điểm chuẩn**",
        lambda h_text, q, h_raw: ask_mysql(h_text, q)      # Truyền h_text cho SQL
    ),
    "QUIZ": (
        "🧩 **Trắc nghiệm**",
        lambda h_text, q, h_raw: "Hãy chuyển sang **Tab '🧩 Trắc nghiệm Hướng nghiệp 🧩'** ở phía trên màn hình nhé!"
    ),
    "CHAT": (
        "👋",
        lambda h_text, q, h_raw: "Chào bạn! Tôi là trợ lý ảo tuyển sinh. Tôi có thể tra cứu điểm chuẩn hoặc tư vấn hướng nghiệp cho bạn."
    ),
}

def run_agents(routes: list[str], history_text: str, user_text: str, raw_history: list) -> str:
    valid_routes = [r for r in routes if r in AGENT_MAP]
    if not valid_routes:
        valid_routes = ["CHAT"]

    if len(valid_routes) == 1:
        header, fn = AGENT_MAP[valid_routes[0]]
        return f"{header}:\n\n{fn(history_text, user_text, raw_history)}"

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(valid_routes)) as executor:
        future_to_route = {
            executor.submit(AGENT_MAP[r][1], history_text, user_text, raw_history): r
            for r in valid_routes
        }
        for future in as_completed(future_to_route):
            route = future_to_route[future]
            try:
                results[route] = future.result()
            except Exception as e:
                results[route] = f"❌ Lỗi agent `{route}`: {e}"

    parts = []
    for r in valid_routes:
        header, _ = AGENT_MAP[r]
        parts.append(f"{header}:\n\n{results.get(r, '❌ Không có kết quả.')}")

    return "\n\n---\n\n".join(parts)

def _handle_file_input(files: list, user_text: str) -> str:

    combined_text = ""

    for file_path in files:

        filename = os.path.basename(file_path)

        extracted = extract_text_from_file(file_path)



        if "❌" in extracted or not extracted.strip():

            return (

                f"⚠️ Xin lỗi, tôi không thể đọc được chữ từ file **'{filename}'**.\n"

                f"Chi tiết: {extracted}"

            )

        combined_text += f"\n--- NỘI DUNG: {filename} ---\n{extracted}\n"



    # Không có câu hỏi kèm theo → chỉ xác nhận đã đọc

    if not user_text.strip():

        return "📄 Tôi đã đọc xong tài liệu. Bạn muốn hỏi thông tin gì về nó?"



    # Có câu hỏi → trả lời dựa vào nội dung file

    try:

        llm_doc = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

        doc_chain = (

            ChatPromptTemplate.from_messages([

                ("system",

                 "Bạn là trợ lý đọc tài liệu. "

                 "Hãy trả lời câu hỏi DỰA VÀO DUY NHẤT NỘI DUNG TÀI LIỆU được cung cấp.\n\n"

                 "{context}"),

                ("human", "{question}"),

            ])

            | llm_doc

            | StrOutputParser()

        )

        answer = doc_chain.invoke({"context": combined_text, "question": user_text})

        return f"👁️ **(Đọc file đính kèm)**:\n\n{answer}"

    except Exception as e:

        return f"❌ Lỗi phân tích file: {e}"


def _handle_text_input(user_text: str, current_user_id: int) -> str:
    if not user_text.strip():
        return "Bạn chưa nhập câu hỏi."

    # 1. Kéo lịch sử từ DB
    try:
        raw_db_history = get_user_history(current_user_id)
        # Format cho Router (Text)
        history_text = format_history_for_router(raw_db_history)
        # Format cho RAG (List[Dict])
        history_rag = format_history_for_rag(raw_db_history)
    except Exception as e:
        return f"❌ Lỗi lấy lịch sử: {e}"

    # 2. Router ra quyết định
    try:
        raw_route = router_chain.invoke({
            "question": user_text,
            "history": history_text,
        }).strip().upper()
    except Exception as e:
        return f"❌ Lỗi Router: {e}"

    routes = [r.strip() for r in raw_route.split(",") if r.strip()]
    print(f"📡 Routes {routes} | User ID: {current_user_id}")

    # 3. Phân phối cho Agents
    return run_agents(routes, history_text, user_text, history_rag)


def main_chat_handler(message_dict: dict, history, current_user_id):
    if current_user_id is None:
        return "⚠️ Vui lòng đăng nhập để sử dụng hệ thống."

    user_text: str = message_dict.get("text", "") or ""
    files: list   = message_dict.get("files", []) or []


    if files:
        ans = _handle_file_input(files, user_text) 
    else:
        ans = _handle_text_input(user_text, current_user_id)

    try:
        saved_text = user_text.strip() or "[Người dùng đã gửi tệp tin đính kèm]"
        save_message_and_keep_top_5(current_user_id, saved_text, ans)
    except Exception as e:
        print(f"⚠️ Không thể lưu MySQL: {e}")

    return ans


# ==============================================================================
# 🖥️ 5. GIAO DIỆN (GRADIO)
# ==============================================================================
print("🚀 Đang khởi động hệ thống giao diện UI...")

with gr.Blocks(title="Hệ thống Tuyển sinh", theme=gr.themes.Soft()) as demo:
    user_state = gr.State(None)
    
    gr.Markdown("# 🤖 SUPER AGENT: TƯ VẤN TUYỂN SINH & HƯỚNG NGHIỆP")
    
    auth_pane, username_input, password_input, login_btn, register_btn, auth_message = create_auth_components()

    with gr.Column(visible=False) as app_pane:
        with gr.Row():
            welcome_msg = gr.Markdown("### Chào mừng bạn trở lại!")
            logout_btn = gr.Button("🚪 Đăng xuất", size="sm", variant="stop")
            
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
                    additional_inputs=[user_state],
                    examples=[
                        [{"text": "Tôi sẽ hợp với nghề gì?"}, None], 
                        [{"text": "Điểm chuẩn ngành Toán Tin Bách Khoa Hà Nội theo THPT năm 2024?"}, None],
                        [{"text": "Các phương thức xét tuyển của Đại học Kinh tế Quốc dân năm 2026?"}, None]
                    ]
                )
            
            with gr.TabItem("🧩 Trắc nghiệm Hướng nghiệp🧩"):
                render_quiz_tab()

    bind_auth_events(
        login_btn, register_btn, username_input, password_input, 
        auth_message, auth_pane, app_pane, user_state, welcome_msg, custom_chatbot_component
    )

    def logout_action():
        return gr.update(visible=True), gr.update(visible=False), None, []

    logout_btn.click(
        logout_action, 
        inputs=[], 
        outputs=[auth_pane, app_pane, user_state, custom_chatbot_component]
    )

if __name__ == "__main__":
    demo.launch(share=False)
