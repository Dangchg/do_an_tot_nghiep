import gradio as gr
import json

class CareerAgent:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.questions = self.data.get('questions', [])
        # SỬA: Lấy dữ liệu từ key 'careerMapping'
        self.career_mapping = self.data.get('careerMapping', {}) 
    
    def load_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            return {}

    def calculate_result(self, user_scores):
        if not user_scores:
            return "Chưa xác định", "Bạn chưa hoàn thành bài trắc nghiệm."

        # Sắp xếp điểm số từ cao xuống thấp
        sorted_scores = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Lấy loại tính cách có điểm cao nhất (Top 1)
        top_type = sorted_scores[0][0]  # Ví dụ: "Realistic"
        
        # Lấy nội dung mô tả từ careerMapping
        result_text = self.career_mapping.get(top_type, "Không tìm thấy thông tin cho nhóm này.")
        
        return top_type, result_text

# Khởi tạo Agent
agent = CareerAgent('career_data.json')

def quiz_logic(current_index, scores, selected_option_text):
    """
    Logic chính của ứng dụng
    """
    # --- 1. CỘNG ĐIỂM (Nếu đã chọn) ---
    if selected_option_text is not None and current_index < len(agent.questions):
        prev_q = agent.questions[current_index]
        # Tìm option tương ứng với text người dùng chọn
        selected_opt = next((opt for opt in prev_q['options'] if opt['text'] == selected_option_text), None)
        
        if selected_opt:
            # Lấy 'type' để tính điểm (VD: Enterprising)
            type_key = selected_opt.get('type')
            if type_key:
                scores[type_key] = scores.get(type_key, 0) + 1
        
        current_index += 1

    # --- 2. KIỂM TRA KẾT THÚC ---
    if current_index >= len(agent.questions):
        top_type, result_content = agent.calculate_result(scores)
        
        # Format hiển thị kết quả đẹp mắt
        result_md = f"""
        # 🎯 KẾT QUẢ ĐỊNH HƯỚNG NGHỀ NGHIỆP
        
        ### Nhóm tính cách nổi bật: {top_type}
        
        ---
        
        ### 📝 Chi tiết:
        
        **{result_content}**
        
        ---
        *Kết quả này dựa trên mô hình trắc nghiệm Holland.*
        """
        
        return (
            current_index, scores, 
            gr.update(visible=False), # Ẩn câu hỏi
            gr.update(visible=False), # Ẩn radio
            gr.update(visible=False), # Ẩn nút Next
            gr.update(visible=True, value=result_md) # Hiện kết quả
        )

    # --- 3. HIỂN THỊ CÂU HỎI TIẾP THEO ---
    next_q = agent.questions[current_index]
    
    # Lấy nội dung câu hỏi
    q_content = next_q.get('question', 'Câu hỏi không có nội dung')
    
    display_text = f"### Câu hỏi {current_index + 1}/{len(agent.questions)}: \n\n {q_content}"
    options_list = [opt['text'] for opt in next_q['options']]
    
    return (
        current_index, scores,
        gr.update(value=display_text, visible=True),
        gr.update(choices=options_list, value=None, visible=True),
        gr.update(visible=True),
        gr.update(visible=False)
    )

# --- GIAO DIỆN DEMO ---
'''with gr.Blocks(title="Tư vấn hướng nghiệp", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🎓 Ứng dụng Tư vấn Hướng nghiệp AI")
    
    state_index = gr.State(0)
    state_scores = gr.State({})
    
    with gr.Column():
        q_display = gr.Markdown("Bấm bắt đầu...")
        opt_radio = gr.Radio(label="Lựa chọn của bạn", interactive=True)
        btn_next = gr.Button("Tiếp tục ➡", variant="primary")
        res_display = gr.Markdown(visible=False)

    btn_next.click(
        quiz_logic,
        [state_index, state_scores, opt_radio],
        [state_index, state_scores, q_display, opt_radio, btn_next, res_display]
    )
    
    # Tự động chạy khi mở app
    demo.load(
        quiz_logic,
        [state_index, state_scores, gr.State(None)],
        [state_index, state_scores, q_display, opt_radio, btn_next, res_display]
    )

if __name__ == "__main__":
    demo.launch()'''

# --- GIAO DIỆN --- 

def render_quiz_tab():
    """Hàm này sẽ được gọi bên trong App chính để vẽ giao diện"""
    with gr.Column():
        gr.Markdown("## 🧩 Bài trắc nghiệm định hướng nghề nghiệp (Holland Code)")
        
        # State
        state_index = gr.State(0)
        state_scores = gr.State({})
        
        # UI Components
        q_display = gr.Markdown("Bấm nút **Bắt đầu** bên dưới...")
        opt_radio = gr.Radio(label="Lựa chọn", interactive=True, visible=False)
        btn_next = gr.Button("Bắt đầu làm bài ➡", variant="primary")
        res_display = gr.Markdown(visible=False)

        # Logic Button
        # Logic đặc biệt cho nút Start/Next để load câu đầu tiên hoặc câu tiếp theo
        def on_click(idx, scr, opt):
            # Nếu là lần đầu (idx=0, opt=None) -> Load câu 1
            # Nếu đang làm -> Gọi quiz_logic
            return quiz_logic(idx, scr, opt)

        btn_next.click(
            quiz_logic,
            inputs=[state_index, state_scores, opt_radio],
            outputs=[state_index, state_scores, q_display, opt_radio, btn_next, res_display]
        )