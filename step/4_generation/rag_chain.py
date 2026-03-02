"""
Bước 3: RAG Chain - Tích hợp LLM + Retriever + Prompts
Sử dụng Ollama với streaming + timing + improved prompt
"""

import os
import time
from typing import Optional, Callable
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from system_prompt import REFUSAL_RESPONSE
from refusal_and_citations import REFUSAL_MESSAGES

# Ollama configuration
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Improved prompt - TRÍCH NGUYÊN VĂN, không tóm tắt
VIETNAMESE_PROMPT = """NHIỆM VỤ: Tìm và trích nguyên văn nội dung liên quan từ TÀI LIỆU bên dưới để trả lời CÂU HỎI.

QUY TẮC TUYỆT ĐỐI:
- CHỈ dùng TIẾNG VIỆT
- CHỈ lấy thông tin từ TÀI LIỆU được cung cấp, KHÔNG dùng kiến thức bên ngoài
- TRÍCH NGUYÊN VĂN câu chữ trong tài liệu, KHÔNG diễn giải lại
- Nếu tài liệu có định nghĩa → trích nguyên văn định nghĩa đó
- Nếu có nhiều khoản (1., 2., 3., ...) → liệt kê ĐỦ TẤT CẢ
- Nếu không tìm thấy trong tài liệu → chỉ nói: "Không tìm thấy thông tin này trong các văn bản được cung cấp."

TÀI LIỆU:
{context}

CÂU HỎI: {query}

TRẢ LỜI (trích nguyên văn từ tài liệu, bằng tiếng Việt):
"""


def load_faiss_vectorstore():
    """Load FAISS index từ giai đoạn 2"""
    print("📦 Loading FAISS index...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vectorstore = FAISS.load_local(
        "step/2_ingestion/output/law_documents_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ Loaded {vectorstore.index.ntotal} vectors")
    return vectorstore


def build_rag_chain(temperature=0.1, top_k=5, rebuild_llm=False):
    """Build RAG chain: Retriever + LLM + Prompt
    
    Args:
        temperature: LLM temperature (0.0-1.0), lower = more factual
        top_k: Number of documents to retrieve in search_kwargs
        rebuild_llm: Force rebuild LLM
    """
    print("🔧 Building RAG chain...")
    
    # 1. Load vectorstore
    vectorstore = load_faiss_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # 2. Init LLM với Ollama
    print(f"[Using Ollama model: {OLLAMA_MODEL}]")
    
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        system="Bạn là trợ lý pháp luật Việt Nam. Bạn PHẢI trả lời HOÀN TOÀN bằng TIẾNG VIỆT. TUYỆT ĐỐI KHÔNG được dùng tiếng Anh trong câu trả lời.",
    )
    
    # 3. Build custom chain
    class CustomRAGChain:
        def __init__(self, llm, retriever, vectorstore, template):
            self.llm = llm
            self.retriever = retriever
            self.vectorstore = vectorstore  # Store for query_rag
            self.template = template
        
        def __call__(self, inputs):
            query = inputs.get("query") or inputs.get("input", "")
            
            # Retrieve documents
            docs = self.retriever.invoke(query)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create full prompt
            full_prompt = self.template.format(context=context, query=query)
            
            # Get answer from LLM
            result = self.llm.invoke(full_prompt)
            answer = result.content if hasattr(result, 'content') else str(result)
            
            return {
                "result": answer,
                "source_documents": docs
            }
        
        def invoke(self, inputs):
            """Alias for __call__"""
            return self(inputs)
    
    qa_chain = CustomRAGChain(llm, retriever, vectorstore, VIETNAMESE_PROMPT)
    
    print("✅ RAG chain built successfully")
    return qa_chain


def query_rag(qa_chain, question: str, max_retries=3, stream_callback: Optional[Callable] = None) -> dict:
    """
    Query RAG chain với streaming và timing
    """
    total_start = time.time()
    
    # ============ CƠ CHẾ TỪ CHỐI - STEP 0 ============
    question_lower = question.lower()
    
    # Từ khóa out-of-domain - từ chối ngay
    out_of_domain_keywords = [
        "who are you", "ai la ai", "ban la ai", "maye la ai", "mai la ai", 
        "ban ten la gi", "ban la gì", "ai la ban", "tim ban",
        "what is your name", "who built you", "tu tien huy",
        "recipe", "nau an", "lam an", "cong thuc nau an",
        "love", "dating", "em la ai", "yeu", "hen ho",
        "joke", "tro chuyen", "tao la ai", "co la ai",
    ]
    
    if any(keyword in question_lower for keyword in out_of_domain_keywords):
        return {
            "answer": REFUSAL_MESSAGES["out_of_scope"],
            "sources": [],
            "source_citations": [],
            "refused": True,
            "timing": {"search_time": 0, "llm_time": 0, "total_time": time.time() - total_start}
        }
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # STEP 1: Tìm kiếm tài liệu
            search_start = time.time()
            print("🔍 Đang tìm kiếm tài liệu...")
            
            docs = qa_chain.retriever.invoke(question)
            search_time = time.time() - search_start
            print(f"✅ Tìm thấy {len(docs)} tài liệu trong {search_time:.2f}s")
            
            # Kiểm tra nếu không tìm thấy tài liệu
            if len(docs) == 0:
                return {
                    "answer": "Không tìm thấy tài liệu liên quan.",
                    "sources": [],
                    "source_citations": [],
                    "refused": True,
                    "timing": {"search_time": search_time, "llm_time": 0, "total_time": time.time() - total_start}
                }
            
            # STEP 2: Tạo prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            full_prompt = qa_chain.template.format(context=context, query=question)
            
            # STEP 3: Generate với streaming
            llm_start = time.time()
            print("🤖 Đang tạo câu trả lời...")
            
            answer_chunks = []
            if stream_callback:
                for chunk in qa_chain.llm.stream(full_prompt):
                    chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    answer_chunks.append(chunk_text)
                    stream_callback(chunk_text)
            else:
                result = qa_chain.llm.invoke(full_prompt)
                answer_chunks = [result.content if hasattr(result, 'content') else str(result)]
            
            answer = "".join(answer_chunks).strip()
            llm_time = time.time() - llm_start
            print(f"✅ Tạo xong trong {llm_time:.2f}s")
            
            # ============ CƠ CHẾ TỪ CHỐI - STEP 4 ============
            # Kiểm traHallucination - nếu trả lời có dấu hiệu bịa đặt
            answer_lower = answer.lower()
            hallucination_indicators = [
                "không có trong tài liệu",
                "không được đề cập",
                "không tìm thấy",
                "thông tin không có",
                "tôi nghĩ", "tôi hiểu",
            ]
            
            # Nếu câu trả lời quá ngắn hoặc có dấu hiệu từ chối
            if not answer or len(answer) < 10:
                return {
                    "answer": "Không tìm thấy thông tin phù hợp trong tài liệu.",
                    "sources": docs,
                    "source_citations": [d.metadata.get("citation", "") for d in docs],
                    "refused": True,
                    "timing": {"search_time": search_time, "llm_time": llm_time, "total_time": time.time() - total_start}
                }
            
            # Trích xuất citations từ metadata
            citations = []
            for doc in docs:
                citation = doc.metadata.get("citation", "")
                if citation and citation not in citations:
                    citations.append(citation)
            
            total_time = time.time() - total_start
            print(f"⏱️ Tổng thời gian: {total_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": docs,
                "source_citations": citations,
                "refused": False,
                "timing": {
                    "search_time": search_time,
                    "llm_time": llm_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[Attempt {attempt + 1}/{max_retries}] Error: {str(e)[:60]}...")
                time.sleep(wait_time)
            else:
                print(f"[Failed after {max_retries} attempts]")
                break
    
    return {
        "answer": f"Lỗi: {str(last_error)}",
        "sources": [],
        "source_citations": [],
        "refused": True,
        "timing": {"search_time": 0, "llm_time": 0, "total_time": time.time() - total_start}
    }


def format_output(result: dict) -> str:
    """Format output cho display"""
    output = []
    output.append("=" * 60)
    
    # Check if answer was refused
    if result.get("refused"):
        output.append("[HE THONG TU CHOI TRA LOI]\n")
        output.append(result["answer"])
    else:
        output.append("TRICH DAN PHAP LUAT\n")
        output.append(result["answer"])
        
        # Only show sources if we have an answer and have citations
        answer = result.get("answer", "").strip()
        if answer and result.get("source_citations"):
            output.append("\nNguon tham khao:")
            for i, citation in enumerate(result["source_citations"], 1):
                output.append(f"{i}. {citation}")
    
    output.append("=" * 60)
    return "\n".join(output)


if __name__ == "__main__":
    print("🚀 Testing RAG chain...\n")
    
    try:
        # Build chain
        qa_chain = build_rag_chain()
        
        # Test query
        test_query = "Quy định về bảo vệ đê điều như thế nào?"
        print(f"❓ Query: {test_query}\n")
        
        result = query_rag(qa_chain, test_query)
        output = format_output(result)
        print(output)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
