# Hướng Dẫn Cài Đặt & Chạy Hệ Thống RAG Legal QA

## Tổng Quan

Hệ thống Q&A về văn bản luật Việt Nam sử dụng RAG (Retrieval-Augmented Generation) với:
- **LLM**: Ollama (llama3) - Chạy local, miễn phí
- **Retrieval**: FAISS + Hybrid Search (BM25 + Dense)
- **Frontend**: Desktop App (Tkinter)

## Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|---------------------|--------------|
| **RAM** | 8 GB | 16 GB |
| **VRAM** | - | 8 GB (nếu dùng GPU) |
| **Disk** | 10 GB | 20 GB |
| **OS** | Windows 10+ | Windows 11 |

---

## Bước 1: Cài Đặt Ollama

### 1.1. Tải Ollama

```bash
# Windows: Download từ https://ollama.com
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com | sh
```

### 1.2. Tải Model llama3

```bash
# Sau khi cài đặt, chạy:
ollama pull llama3

# Kiểm tra model đã tải:
ollama list
```

### 1.3. Khởi động Ollama

```bash
# Chạy Ollama server:
ollama serve

# Hoặc chạy trong background:
# Windows: Ollama sẽ tự khởi động khi cần
```

---

## Bước 2: Cài Đặt Python & Dependencies

### 2.1. Yêu cầu Python

- **Python**: 3.10 hoặc cao hơn
- Kiểm tra: `python --version`

### 2.2. Cài đặt Dependencies

```bash
# Clone hoặc tải source code
cd EnglishforIT2

# Tạo virtual environment (khuyến nghị):
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài dependencies:
pip install -r requirements.txt
```

### 2.3. Nội dung requirements.txt

```
langchain>=0.1.0
langchain-core>=1.2.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
langchain-ollama>=0.1.0
sentence-transformers>=2.0.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
numpy>=1.24.0
python-dotenv>=0.21.0
pyyaml>=5.3.0
```

---

## Bước 3: Cấu Trúc Dự Án

```
EnglishforIT2/
├── data/
│   └── input/
│       ├── luatdedieu.json
│       ├── luatkhituongthuyvan.json
│       ├── luatphongchongthientai.json
│       └── luatthuyloi.json
│
├── step/
│   ├── 1_dataclean/         # Làm sạch dữ liệu
│   ├── 2_ingestion/        # Tạo FAISS index
│   ├── 3_retrieval/        # Hybrid search
│   ├── 4_generation/       # RAG chain (LLM)
│   │   ├── rag_chain.py
│   │   ├── system_prompt.py
│   │   └── refusal_and_citations.py
│   └── 5_demo/            # Desktop app
│       └── desktop_app.py
│
├── requirements.txt
└── README.md
```

---

## Bước 4: Chạy Ứng Dụng

### 4.1. Chạy Desktop App

```bash
# Từ thư mục gốc:
python step/5_demo/desktop_app.py
```

### 4.2. Giao diện Desktop App

- **Tab Câu trả lời**: Hiển thị câu trả lời từ LLM
- **Tab Tài liệu**: Hiển thị nguồn tham khảo
- **Tab Cài đặt**:
  - Temperature: Điều chỉnh độ sáng tạo (0.0-1.0)
  - Top K: Số lượng tài liệu lấy ra (1-15)

---

## Các Tính Năng Chính

### 1. Streaming Response
- Hiển thị câu trả lời từng phần (như ChatGPT)
- Cursor ▌ indicating đang generate

### 2. Timing Display
- Thời gian tìm kiếm tài liệu
- Thời gian tạo câu trả lời
- Tổng thời gian xử lý

### 3. Smart Refusal
- Từ chối câu hỏi ngoài lĩnh vực luật
- Từ chối khi không tìm thấy tài liệu liên quan

### 4. Citation Extraction
- Trích xuất nguồn từ metadata
- Hiển thị số điều, khoản, văn bản luật

---

## Các Lỗi Thường Gặp & Cách Khắc Phục

### Lỗi 1: Ollama not running

```
Error: Connection refused to http://localhost:11434
```

**Giải pháp:**
```bash
# Khởi động Ollama:
ollama serve
```

### Lỗi 2: Model not found

```
Error: model 'llama3' not found
```

**Giải pháp:**
```bash
# Tải model:
ollama pull llama3
```

### Lỗi 3: Out of memory

```
Error: CUDA out of memory / OOM
```

**Giải pháp:**
- Đảm bảo còn ~8GB RAM free
- Giảm top_k trong Settings (1-5)
- Ollama sẽ tự dùng CPU nếu GPU không đủ

### Lỗi 4: LangChain DeprecationWarning

```
LangChainDeprecationWarning: The class 'HuggingFaceEmbeddings' was deprecated
```

**Giải pháp:** Bỏ qua - vẫn hoạt động bình thường

---

## Tùy Chỉnh Nâng Cao

### 1. Đổi Model

Trong `step/4_generation/rag_chain.py`:

```python
# Đổi model khác:
OLLAMA_MODEL = "mistral"  # Model nhẹ hơn
# hoặc
OLLAMA_MODEL = "phi3"    # Model rất nhẹ
```

Sau đó tải model:
```bash
ollama pull mistral
```

### 2. Điều Chỉnh Temperature

- **0.1**: Chính xác, ít sáng tạo (khuyến nghị)
- **0.5**: Cân bằng
- **0.9**: Sáng tạo, có thể hallucinate

### 3. Điều Chỉnh Top K

- **5-10**: Đủ cho hỏi đáp thông thường
- **15**: Lấy nhiều tài liệu hơn (cho câu hỏi cần nhiều context)

---

## Dữ Liệu

### Nguồn Dữ Liệu

- **212 điều luật** từ 4 văn bản:
  - Luật Đê Điều (48 điều)
  - Luật Thủy Lợi (60 điều)
  - Luật Khí Tượng Thủy Văn (57 điều)
  - Luật Phòng Chống Thiên Tai (47 điều)

### Chất Lượng Dữ Liệu

- ✅ 100% unique IDs
- ✅ Citations chính xác
- ✅ Metadata đầy đủ

---

## Đánh Giá Hiệu Năng

| Metric | Giá trị |
|--------|---------|
| Thời gian tìm kiếm | ~0.1-0.5s |
| Thời gian tạo câu trả lời | ~3-10s |
| Số lượng documents | 212 |
| Vector dimension | 384 |

---

## Giấy Phép

MIT License

---

## Liên Hệ

- GitHub: https://github.com/Nguyen15idhue/EnglishforIT
- Email: [your-email@example.com]

---

**Chúc bạn thành công!**
