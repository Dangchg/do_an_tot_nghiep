import os
import re
import glob
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

load_dotenv()
os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Khởi tạo LLM một lần duy nhất (tránh tạo lại mỗi lần gọi)
_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────────────
# 1. PARSE METADATA TỪ ĐƯỜNG DẪN FILE
# ─────────────────────────────────────────────────────────────────────

DOC_TYPE_MAP = {
    "de-an-tuyen-sinh": "de_an_tuyen_sinh",
    "diem-chuan":       "diem_chuan",
    "thong-bao":        "thong_bao",
    "dang":             "dang",
}


def parse_path_metadata(file_path: str) -> dict:
    """
    Trích metadata từ đường dẫn file theo quy ước:
      Data/{truong}/{nam}/{loai_tai_lieu}_{ten_file}.{ext}.{nam}
    """
    p = Path(file_path)
    parts = p.parts  # ('Data', 'hust', '2025', 'de-an-tuyen-sinh_...')

    meta = {
        "source":   str(file_path),
        "truong":   parts[1] if len(parts) > 1 else "unknown",
        "nam":      parts[2] if len(parts) > 2 else "unknown",
        "doc_type": "unknown",
        "file_ext": "",
    }

    suffixes = p.suffixes  # ['.html', '.2025']
    if suffixes:
        last = suffixes[-1].lstrip(".")
        if last.isdigit():
            meta["file_ext"] = suffixes[-2] if len(suffixes) >= 2 else ""
        else:
            meta["file_ext"] = suffixes[-1]

    base_no_ext = p.name.split(".")[0]
    for prefix, dtype in DOC_TYPE_MAP.items():
        if base_no_ext.startswith(prefix):
            meta["doc_type"] = dtype
            break

    return meta


# ─────────────────────────────────────────────────────────────────────
# 2. LLM METADATA ENRICHMENT  (dùng ChatOpenAI đúng cú pháp)
# ─────────────────────────────────────────────────────────────────────

def enrich_metadata_with_llm(text_sample: str, path_meta: dict) -> dict:
    """
    Dùng GPT-4o-mini để nhận dạng thêm thông tin ngữ nghĩa từ nội dung.
    Chỉ truyền 500 ký tự đầu để tiết kiệm token.

    Trả về dict bổ sung (merge vào path_meta):
      - ten_truong_day_du : tên trường đầy đủ
      - loai_hinh_dao_tao : đại học / sau đại học / cao đẳng ...
      - nganh_noi_bat     : danh sách ngành nổi bật (nếu có)
      - chu_de_chinh      : tóm tắt chủ đề chính của đoạn
    """
    prompt = f"""Bạn là hệ thống trích xuất metadata cho tài liệu tuyển sinh đại học Việt Nam.

Thông tin đường dẫn đã có:
- Trường (mã): {path_meta.get('truong')}
- Năm: {path_meta.get('nam')}
- Loại tài liệu (đã đoán): {path_meta.get('doc_type')}

Đoạn nội dung mẫu (500 ký tự đầu):
\"\"\"
{text_sample[:500]}
\"\"\"

Hãy trả về JSON (chỉ JSON, không giải thích, không markdown) với các trường:
{{
  "ten_truong_day_du": "<tên trường đầy đủ hoặc null>",
  "loai_hinh_dao_tao": "<đại học | sau đại học | cao đẳng | khác>",
  "nganh_noi_bat": ["<ngành 1>", "<ngành 2>"],
  "chu_de_chinh": "<mô tả ngắn chủ đề chính của tài liệu>"
}}"""

    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        llm_meta = json.loads(raw)

    except Exception as e:
        print(f"  ⚠️  LLM metadata lỗi ({e}), dùng giá trị mặc định.")
        llm_meta = {
            "ten_truong_day_du": None,
            "loai_hinh_dao_tao": "unknown",
            "nganh_noi_bat":     [],
            "chu_de_chinh":      "",
        }

    return llm_meta


# ─────────────────────────────────────────────────────────────────────
# 3. LOAD & CHUNK DOCUMENTS
# ─────────────────────────────────────────────────────────────────────

def load_file(file_path: str) -> Optional[str]:
    """Đọc nội dung file, hỗ trợ HTML và text."""
    try:
        if ".html" in file_path:
            loader = UnstructuredHTMLLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)
    except Exception as e:
        print(f"  ⚠️  Không đọc được {file_path}: {e}")
        return None


def load_data_from_folder(
    root: str = "Data",
    use_llm_enrichment: bool = True,
    llm_sample_limit: int = 100,
) -> list[Document]:
    """
    Quét toàn bộ Data/**/* , trích metadata từ đường dẫn,
    tùy chọn làm giàu bằng LLM, rồi chunk và trả về danh sách Document.
    """
    all_files = [f for f in glob.glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)]
    print(f"🔍 Tìm thấy {len(all_files)} file trong '{root}'")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    llm_call_count = 0

    for file_path in all_files:
        path_meta = parse_path_metadata(file_path)

        content = load_file(file_path)
        if not content or len(content.strip()) < 30:
            continue

        llm_meta = {}
        if use_llm_enrichment and llm_call_count < llm_sample_limit:
            print(f"  🤖 LLM enriching: {Path(file_path).name}")
            llm_meta = enrich_metadata_with_llm(content, path_meta)
            llm_call_count += 1

        full_meta = {**path_meta, **llm_meta}

        # Chroma chỉ lưu được string/int/float → flatten list
        if isinstance(full_meta.get("nganh_noi_bat"), list):
            full_meta["nganh_noi_bat"] = ", ".join(full_meta["nganh_noi_bat"])

        base_doc = Document(page_content=content, metadata=full_meta)
        chunks = text_splitter.split_documents([base_doc])
        all_chunks.extend(chunks)

    doc_types = set(c.metadata.get("doc_type", "?") for c in all_chunks)
    truongs   = set(c.metadata.get("truong",   "?") for c in all_chunks)
    print(f"✅ Tổng chunks: {len(all_chunks)}")
    print(f"   Loại tài liệu : {', '.join(doc_types)}")
    print(f"   Trường        : {', '.join(truongs)}")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────
# 4. BUILD / LOAD VECTOR STORE
# ─────────────────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: list[Document],
    db_name: str = "vector_db",
    device: str = "cuda",
    rebuild: bool = True,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32},
    )

    if rebuild and os.path.exists(db_name):
        print("🗑️  Xóa vector DB cũ...")
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    print("⚙️  Đang tạo vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name,
    )
    n = vectorstore._collection.count()
    sample_emb = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    print(f"✅ Vector store: {n} vectors, {len(sample_emb):,} chiều")
    return vectorstore


def load_vectorstore(db_name: str = "vector_db", device: str = "cpu") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs={"device": device},
    )
    print(f"📂 Load Vector DB từ '{db_name}'...")
    return Chroma(persist_directory=db_name, embedding_function=embeddings)


# ─────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def build_chroma_filter(filter_meta: Optional[dict]) -> Optional[dict]:
    """
    Chuyển dict filter thông thường sang cú pháp Chroma:
      - 1 điều kiện  → {"key": {"$eq": "value"}}
      - nhiều điều kiện → {"$and": [{"key": {"$eq": "value"}}, ...]}
    """
    if not filter_meta:
        return None
    conditions = [{k: {"$eq": v}} for k, v in filter_meta.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def get_retriever(vectorstore: Chroma, k: int = 5, filter_meta: Optional[dict] = None):
    search_kwargs = {"k": k}
    chroma_filter = build_chroma_filter(filter_meta)
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def rag_query(
    query: str,
    vectorstore: Chroma,
    k: int = 5,
    filter_meta: Optional[dict] = None,
) -> list[Document]:
    retriever = get_retriever(vectorstore, k=k, filter_meta=filter_meta)
    docs = retriever.invoke(query)
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(
            f"  [{i}] {m.get('truong','?')} | {m.get('nam','?')} "
            f"| {m.get('doc_type','?')} | {m.get('ten_truong_day_du','?')}"
        )
    return docs


# ─────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

def visualize_vectorstore(vectorstore: Chroma, color_by: str = "doc_type"):
    result = vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
    vectors   = np.array(result["embeddings"])
    documents = result["documents"]
    metadatas = result["metadatas"]

    labels = [m.get(color_by, "unknown") for m in metadatas]
    unique_labels = sorted(set(labels))
    palette = ["blue", "red", "green", "orange", "purple", "brown", "pink"]
    label_color = {l: palette[i % len(palette)] for i, l in enumerate(unique_labels)}
    colors = [label_color[l] for l in labels]

    tsne = TSNE(n_components=2, random_state=42)
    rv   = tsne.fit_transform(vectors)

    hover = [
        f"Trường: {m.get('truong','?')}<br>"
        f"Năm: {m.get('nam','?')}<br>"
        f"Loại: {m.get('doc_type','?')}<br>"
        f"Chủ đề: {m.get('chu_de_chinh','')}<br>"
        f"Nội dung: {d[:120]}..."
        for m, d in zip(metadatas, documents)
    ]

    fig = go.Figure(data=[go.Scatter(
        x=rv[:, 0], y=rv[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.8),
        text=hover,
        hoverinfo="text",
    )])
    fig.update_layout(
        title=f"2D Vector Store — tô màu theo '{color_by}'",
        width=900, height=650,
        margin=dict(r=20, b=10, l=10, t=40),
    )
    fig.show(renderer="browser")


# ─────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REBUILD = False  # True: tạo lại DB | False: load DB đã có

    if REBUILD:
        chunks = load_data_from_folder(root="Data", use_llm_enrichment=True, llm_sample_limit=50)
        vs = build_vectorstore(chunks, db_name="vector_db", device="cuda", rebuild=True)
    else:
        vs = load_vectorstore(db_name="vector_db", device="cpu")

    visualize_vectorstore(vs, color_by="doc_type")

    print("\n--- RAG query ví dụ ---")

    # 1 điều kiện — filter đơn giản
    results = rag_query(
        query="Xét tuyển tài năng 2025 bưu chính viễn thông",
        vectorstore=vs,
        k=5,
        filter_meta={"truong": "neu"},
    )

    # Nhiều điều kiện — tự động dùng $and
    results = rag_query(
        query="Các ngành tuyển sinh năm 2025",
        vectorstore=vs,
        k=5,
        filter_meta={"truong": "hust", "nam": "2025"},
    )
