import subprocess
import requests
import time
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN OCR
# ==========================================
# 1. Đường dẫn tới thư mục chứa code OCR (api_server.py)
OCR_PROJECT_DIR = r"C:\Users\Admin\Desktop\OCR\api_server.py"

# 2. Đường dẫn tới file python.exe trong môi trường ảo (venv) của dự án OCR
# Ví dụ Windows: C:\path\to\ocr_project\venv\Scripts\python.exe
# Ví dụ Mac/Linux: /path/to/ocr_project/venv/bin/python
OCR_PYTHON_ENV = r"C:\Users\Admin\Desktop\OCR\venv\Scripts\python.exe" 

# 3. Thông tin API
OCR_API_URL = "http://127.0.0.1:8000/api/v1/extract-text"
OCR_PROCESS = None # Biến toàn cục lưu trạng thái tiến trình

def is_ocr_server_running():
    """Kiểm tra xem API Server OCR đã sẵn sàng chưa bằng cách gửi 1 request nhẹ"""
    try:
        # FastAPI mặc định có endpoint /docs, dùng nó để ping kiểm tra
        res = requests.get("http://127.0.0.1:8000/docs", timeout=2)
        return res.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ocr_server():
    """Khởi động API Server OCR từ một môi trường khác"""
    global OCR_PROCESS
    
    if is_ocr_server_running():
        print("✅ [OCR Agent] Server OCR đã đang chạy!")
        return True

    print("🚀 [OCR Agent] Đang khởi động Server OCR ở môi trường độc lập...")
    
    try:
        # Lệnh chạy: python api_server.py
        # Tham số cwd (current working directory) rất quan trọng để mô hình load đúng đường dẫn tương đối
        OCR_PROCESS = subprocess.Popen(
            [OCR_PYTHON_ENV, "api_server.py"],
            cwd=OCR_PROJECT_DIR,
            stdout=subprocess.PIPE, # Ẩn log của server OCR khỏi console hiện tại
            stderr=subprocess.PIPE
        )
        
        # Đợi tối đa 20 giây để server khởi động (vì load AI model thường khá lâu)
        max_wait = 20
        print("⏳ Đang nạp mô hình AI OCR, vui lòng đợi (Tối đa 20s)...")
        for i in range(max_wait):
            if is_ocr_server_running():
                print("✅ [OCR Agent] Khởi động Server OCR thành công!")
                return True
            time.sleep(1)
            
        print("❌ [OCR Agent] Timeout! Server OCR không phản hồi sau 20s.")
        return False
        
    except Exception as e:
        print(f"❌ [OCR Agent] Lỗi khi gọi subprocess: {e}")
        return False

def extract_text_via_ocr_api(file_path):
    """Gửi file ảnh/PDF sang Server OCR và nhận text về"""
    # 1. Đảm bảo server đang chạy
    if not start_ocr_server():
        return "❌ Lỗi: Không thể kết nối với Hệ thống OCR cục bộ."
        
    # 2. Gửi request
    print(f"📡 [OCR Agent] Đang gửi '{os.path.basename(file_path)}' sang Server OCR để quét...")
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/pdf")} # FastAPI xử lý được cả ảnh và PDF
            # Tăng timeout lên 60s vì PDF nhiều trang có thể xử lý lâu
            response = requests.post(OCR_API_URL, files=files, timeout=1000)
            
        if response.status_code == 200:
            data = response.json()
            return data["text"]
        else:
            return f"❌ [OCR Agent] Lỗi từ Server OCR: {response.text}"
            
    except Exception as e:
        return f"❌ [OCR Agent] Lỗi truyền dữ liệu: {str(e)}"

# Gọi hàm này khi thoát ứng dụng chính để tắt luôn Server OCR (nếu muốn)
def stop_ocr_server():
    global OCR_PROCESS
    if OCR_PROCESS:
        OCR_PROCESS.terminate()
        print("🛑 [OCR Agent] Đã tắt Server OCR.")
