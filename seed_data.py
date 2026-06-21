# imports
import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def load_data_from_folder(folders = glob.glob("Data/*")):
        
    text_loader_kwargs={'autodetect_encoding': True}
    metadata_sr = []
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    print("Total documents loaded:", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Slightly smaller chunks for better retrieval
        chunk_overlap=50,  # Reduced overlap for performance
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separation
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Các loại tài liệu đã tìm thấy: {', '.join(doc_types)}")

    return chunks
    

def vector_store1(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        model_kwargs={"device": "cuda"},   # 👈 dùng GPU
        encode_kwargs={"batch_size": 32}   # 👈 tăng batch để tận dụng GPU
    )

    # Đặt tên cho database vector (có thể tùy chọn)
    db_name = "vector_db"

    # Kiểm tra nếu database Chroma đã tồn tại, thì xóa collection để khởi động lại từ đầu
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    # Tạo vector store bằng Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,              # Danh sách các đoạn văn bản đã chia nhỏ
        embedding=embeddings,          # Hàm embedding (ví dụ: OpenAI hoặc HuggingFace)
        persist_directory=db_name      # Thư mục lưu trữ database
    )
    # Kiểm tra số lượng document đã được lưu vào vector store
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    # Lấy ra bộ sưu tập vector từ vectorstore
    collection = vectorstore._collection

    # Lấy 1 embedding từ database
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]

    # Kiểm tra số chiều (số phần tử trong vector)
    dimensions = len(sample_embedding)
    print(f"The vectors have {dimensions:,} dimensions")
    # Lấy toàn bộ vector, tài liệu và metadata từ collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])

    # Đưa embedding vào mảng numpy
    vectors = np.array(result['embeddings'])

    # Lưu lại văn bản
    documents = result['documents']

    # Trích loại tài liệu từ metadata (giả sử có 'doc_type')
    doc_types = [metadata['doc_type'] for metadata in result['metadatas']]

    # Gán màu sắc tùy theo loại tài liệu
    colors = [['blue', 'green', 'red'][['Dang', 'de_an_tuyen_sinh', 'diem_chuan'].index(t)] for t in doc_types]

    # 2D dimension!
    # Giảm số chiều của vector xuống 2D bằng t-SNE
    # (T-distributed Stochastic Neighbor Embedding)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Tạo biểu đồ scatter 2D
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Loại: {t}<br>Văn bản: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='Biểu đồ 2D Chroma Vector Store',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show(renderer="browser")
    
    return vectorstore
def vector_store2(chunks):
    db_name = "vector_db"
    embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding_v2")
    if os.path.exists(db_name):  
        print("📂 Đang load Vector DB (đã lưu trước đó)...")
        vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        )
    return vectorstore
