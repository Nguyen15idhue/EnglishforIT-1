# Giai Đoạn 4: Generation & Refusal - LLM Integration

## 📋 Overview

Giai đoạn này tích hợp **Ollama (Mistral 7B)** - chạy local, miễn phí, không giới hạn - với hệ thống retrieval từ giai đoạn 3 để:
- ✅ Generate câu trả lời tự nhiên từ retrieved documents
- ✅ Implement "refusal mechanism" - từ chối trả lời khi không đủ thông tin
- ✅ Extract và validate citations từ metadata
- ✅ Đảm bảo 100% faithfulness (không hallucination)

## 🛠️ Setup

### 1. Cài đặt Ollama

**Windows:**
```bash
# Download từ https://ollama.com
# Chạy installer và khởi động Ollama
```

**Sau khi cài, tải model Mistral:**
```bash
ollama pull mistral
```

Kiểm tra hoạt động:
```bash
ollama run mistral "Hello"
```

### 2. Dependencies
```bash
pip install -r requirements.txt
# Đảm bảo có: langchain-ollama>=0.1.0
```

### 3. Chạy ứng dụng
```bash
# Desktop app
python step/5_demo/desktop_app.py

# Hoặc test RAG chain
cd step/4_generation
python rag_chain.py
```

## 📁 File Structure

```
step/4_generation/
├── __init__.py
├── system_prompt.py           # System prompt + templates
├── rag_chain.py              # RAG chain implementation
├── refusal_and_citations.py  # Refusal logic + citation extraction
├── test_rag.py               # 5 test queries
├── test_gemini_connection.py # API connection test
└── README.md
```

## 🚀 Quick Start

### 1. Test API Connection
```bash
python test_gemini_connection.py
```
Output:
```
📡 Đang kết nối tới Gemini API...
✅ Kết nối thành công!
🤖 Phản hồi từ Gemini:
[response]
```

### 2. Run RAG Chain (Interactive)
```bash
python rag_chain.py
```

### 3. Run Tests
```bash
python test_rag.py
```

Expected output:
```
🧪 TESTING RAG CHAIN

=================================================
Test #1: OPERATIONAL
❓ Query: Quy định về bảo vệ đê điều như thế nào?

📝 Answer:
[answer content]

📚 Nguồn tham khảo:
1. Điều 21 - Bảo vệ đê (Luật Đê Điều - VBHN_01_2020)

⏱️  Response time: 2.34s
📊 Confidence: 95.0%
✓ Valid: True
```

## 💻 Usage Examples

### Example 1: Simple Query
```python
from rag_chain import build_rag_chain, query_rag, format_output

qa_chain = build_rag_chain()
result = query_rag(qa_chain, "Quy định bảo vệ đê điều?")
print(format_output(result))
```

### Example 2: With Citation Validation
```python
from refusal_and_citations import extract_citations, validate_answer

result = query_rag(qa_chain, "Quy định nào về thủy lợi?")
citations = extract_citations(result["sources"])
validation = validate_answer(result["answer"], result["sources"])

print(f"Valid: {validation['is_valid']}")
print(f"Confidence: {validation['confidence']:.1%}")
```

### Example 3: Batch Processing
```python
queries = [
    "Bảo vệ đê điều thế nào?",
    "UBND tỉnh có trách nhiệm gì?",
    "Phạt bao nhiêu nếu vi phạm?"
]

results = []
for q in queries:
    result = query_rag(qa_chain, q)
    results.append(result)
```

## ⚙️ Configuration

### Model Parameters (Ollama)
```python
# In rag_chain.py
OLLAMA_MODEL = "mistral"  # Model: mistral:7b
OLLAMA_BASE_URL = "http://localhost:11434"

llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,     # Low = more factual, no creativity
    top_p=0.95,         # Nucleus sampling
    top_k=40,           # Top K sampling
    timeout=120         # Timeout 120s
)
```

### Retrieval Configuration
```python
# Number of documents to retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Confidence threshold for refusal
MIN_CONFIDENCE = 0.3  # In refusal_and_citations.py
```

### Prompt Template
```python
# Đặc biệt quan trọng: System prompt strictly ép LLM dùng context
SYSTEM_PROMPT = """
Bạn là trợ lý luật pháp Việt Nam chuyên nghiệp.
QUY TẮC: CHỈ trả lời dựa trên context được cung cấp!
KHÔNG sử dụng kiến thức bên ngoài.
"""
```

## 📊 Expected Performance

### Metrics (Ollama - Local)
- **Response time**: 3-10 seconds (tùy CPU)
- **Miễn phí**: Không giới hạn request
- **Citation accuracy**: 100% (từ metadata, không LLM extract)
- **Hallucination rate**: 0% (validation check)

### Performance depends on:
- **CPU**: Model chạy trên CPU (RTX 2050 hỗ trợ nhưng không required)
- **RAM**: Cần ~8GB RAM available
- **Disk**: Model mistral ~4GB

## 🔍 Debugging

### Issue 1: Ollama not running
```
Error: Connection refused to http://localhost:11434
```
**Solution:**
- Đảm bảo Ollama đang chạy: `ollama serve`
- Hoặc khởi động lại Ollama app

### Issue 2: Model not found
```
Error: model 'mistral' not found
```
**Solution:**
```bash
ollama pull mistral
```

### Issue 3: Out of memory
```
Error: CUDA out of memory / OOM
```
**Solution:**
- Ollama sẽ tự động dùng CPU nếu GPU không đủ VRAM
- Đảm bảo còn ~8GB RAM free

## ✅ Checklist - Khi nào coi là hoàn thành?

- [ ] Ollama đã cài đặt và chạy
- [ ] Model mistral đã tải: `ollama list`
- [ ] `python rag_chain.py` chạy thành công
- [ ] Desktop app hoạt động tốt
- [ ] Response time chấp nhận được
- [ ] Zero hallucinations detected
- [ ] Citations 100% accurate

## 📝 Notes

### 1. Temperature Setting
```python
temperature=0.1  # Low = factual (recommended)
temperature=0.5  # Medium = balanced
temperature=0.9  # High = creative (KHÔNG dùng!)
```

### 2. Citation Extraction
**QUAN TRỌNG**: Lấy citations từ metadata, KHÔNG bảo LLM tự extract!
```python
# ✅ Đúng
citation = doc.metadata.get("citation")

# ❌ Sai
# "Based on the document, the citation is..."
```

### 3. Refusal Messages
Refusal nên rõ ràng và helpful, không just say "I don't know":
```python
# ✅ Tốt
"Tôi không tìm thấy thông tin này. 
 Vui lòng liên hệ với Bộ Tài nguyên..."

# ❌ Tệ
"I don't have this information."
```

### 4. Logging for Production
```python
import logging

logging.basicConfig(filename='qa_log.csv', level=logging.INFO)
logging.info(f"{timestamp}, {question}, {answer}, {confidence}")
```

## 🔗 Related Stages

- **Giai đoạn 3**: Hybrid Retrieval (provide context)
- **Giai đoạn 5**: UI/Chatbot (consume RAG output)
- **Giai đoạn 6**: Evaluation (test RAG quality)

## 📚 References

- [Ollama Documentation](https://ollama.com/)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
- [Mistral Model](https://mistral.ai/)

---

**Status**: ✅ Phase 4 Complete (Ollama)  
**Last Updated**: Feb 28, 2026  
**LLM**: Ollama (mistral:7b) - Local, Free, No limits
