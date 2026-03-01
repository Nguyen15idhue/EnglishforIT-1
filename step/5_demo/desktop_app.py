"""
Vietnamese Legal Assistant - Desktop GUI (Tkinter)
Chạy trực tiếp mà không cần localhost
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import os
from datetime import datetime
import threading

# Add path to import rag_chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '4_generation'))

from rag_chain import build_rag_chain, query_rag

# ============================================================================
# APP CLASS
# ============================================================================

class LegalAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚖️ Trợ lý Pháp luật Việt Nam")
        self.root.geometry("1000x850")
        self.root.resizable(True, True)
        
        # State
        self.qa_chain = None
        self.query_history = []
        self.is_loading = False
        self.current_answer_chunks = []
        
        # Colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.button_color = "#667eea"
        self.accent_color = "#764ba2"
        
        self.root.configure(bg=self.bg_color)
        
        # Build UI
        self.build_ui()
        
        # Load RAG chain on startup
        self.load_chain()
    
    def build_ui(self):
        """Tạo giao diện"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=15, pady=10)
        
        title = tk.Label(
            header_frame,
            text="⚖️ TRỢ LÝ PHÁP LUẬT VIỆT NAM",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.button_color
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Hệ thống Q&A thông minh về văn bản luật - RAG + LLAMA3 (Mistral 7B) - Tiếng Việt",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg="#999"
        )
        subtitle.pack()
        
        # Separator
        sep1 = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep1.pack(fill=tk.X, padx=10, pady=5)
        
        # Question section
        question_label = tk.Label(
            self.root,
            text="❓ Đặt câu hỏi:",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        question_label.pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        self.question_input = scrolledtext.ScrolledText(
            self.root,
            height=3,
            font=("Segoe UI", 10),
            bg="#2d2d2d",
            fg=self.fg_color,
            insertbackground=self.button_color
        )
        self.question_input.pack(fill=tk.BOTH, padx=15, pady=(0, 10))
        self.question_input.insert(1.0, "Ví dụ: Luật Thủy lợi quy định gì về bảo vệ công trình nước?")
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg=self.bg_color)
        button_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.search_btn = tk.Button(
            button_frame,
            text="🔍 Tìm kiếm",
            font=("Segoe UI", 10, "bold"),
            bg=self.button_color,
            fg=self.fg_color,
            padx=20,
            pady=8,
            command=self.search
        )
        self.search_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Xóa",
            font=("Segoe UI", 10),
            bg="#555",
            fg=self.fg_color,
            padx=15,
            pady=8,
            command=self.clear
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = tk.Button(
            button_frame,
            text="❌ Thoát",
            font=("Segoe UI", 10),
            bg="#d9534f",
            fg=self.fg_color,
            padx=15,
            pady=8,
            command=self.root.quit
        )
        exit_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            button_frame,
            text="🟢 Sẵn sàng",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#60d66d"
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Separator
        sep2 = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=10, pady=5)
        
        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Tab 1: Answer
        tab1 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab1, text="📝 Câu trả lời")
        
        self.answer_output = scrolledtext.ScrolledText(
            tab1,
            font=("Segoe UI", 10),
            bg="#2d2d2d",
            fg=self.fg_color,
            state=tk.DISABLED
        )
        self.answer_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 2: Sources
        tab2 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab2, text="📚 Tài liệu liên quan")
        
        self.sources_output = scrolledtext.ScrolledText(
            tab2,
            font=("Segoe UI", 9),
            bg="#2d2d2d",
            fg=self.fg_color,
            state=tk.DISABLED
        )
        self.sources_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 3: Settings
        tab3 = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(tab3, text="⚙️ Cài đặt")
        
        settings_frame = tk.Frame(tab3, bg=self.bg_color)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Temperature
        temp_label = tk.Label(
            settings_frame,
            text="Nhiệt độ LLM (Factuality):",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        temp_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.temperature_var = tk.DoubleVar(value=0.1)
        self.temp_slider = tk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.temperature_var,
            bg=self.button_color,
            fg=self.fg_color,
            length=300
        )
        self.temp_slider.pack(anchor=tk.W, pady=(0, 5))
        
        temp_hint = tk.Label(
            settings_frame,
            text="0.0 = Chính xác  |  1.0 = Sáng tạo",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#999"
        )
        temp_hint.pack(anchor=tk.W, pady=(0, 20))
        
        # Top K
        topk_label = tk.Label(
            settings_frame,
            text="Số lượng tài liệu lấy ra:",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        topk_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.topk_var = tk.IntVar(value=10)  # Tăng lên 10 để lấy đủ tài liệu
        self.topk_slider = tk.Scale(
            settings_frame,
            from_=1,
            to=15,
            orient=tk.HORIZONTAL,
            variable=self.topk_var,
            bg=self.button_color,
            fg=self.fg_color,
            length=300
        )
        self.topk_slider.pack(anchor=tk.W, pady=(0, 5))
        
        topk_hint = tk.Label(
            settings_frame,
            text="1 = Ít  |  15 = Nhiều (Dùng 10-15 để lấy đủ thông tin)",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#999"
        )
        topk_hint.pack(anchor=tk.W)
        
        # Separator
        sep3 = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep3.pack(fill=tk.X, padx=10, pady=5)
        
        # Footer info
        footer_frame = tk.Frame(self.root, bg=self.bg_color, height=80)
        footer_frame.pack(fill=tk.X, padx=15, pady=10)
        
        info_text = "📊 Thông tin: 212 điều luật | RAG + Ollama (Mistral 7B) | Tiếng Việt"
        info_label = tk.Label(
            footer_frame,
            text=info_text,
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#999"
        )
        info_label.pack(anchor=tk.W)
    
    def load_chain(self):
        """Load RAG chain in background"""
        def _load():
            try:
                self.status_label.config(text="🔄 Khởi tạo (20-30s)...", fg="#ffc107")
                self.root.update()
                
                self.qa_chain = build_rag_chain(temperature=0.1, top_k=10)
                
                self.status_label.config(text="✅ Sẵn sàng", fg="#60d66d")
            except Exception as e:
                self.status_label.config(text="❌ Lỗi", fg="#d9534f")
                messagebox.showerror("Lỗi", f"Không thể khởi tạo:\n\n{str(e)}")
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
    
    def search(self):
        """Search/Query với streaming và timing"""
        if self.is_loading:
            messagebox.showinfo("Chờ một chút", "Đang xử lý câu hỏi trước...")
            return
        
        if not self.qa_chain:
            messagebox.showwarning("Chưa sẵn sàng", "Hệ thống đang khởi tạo, vui lòng chờ...")
            return
        
        question = self.question_input.get("1.0", tk.END).strip()
        
        if not question or question == "Ví dụ: Luật Thủy lợi quy định gì về bảo vệ công trình nước?":
            messagebox.showwarning("⚠️ Lỗi", "Vui lòng nhập câu hỏi!")
            return
        
        # Add to history
        self.query_history.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {question[:50]}...")
        
        # Reset streaming
        self.current_answer_chunks = []
        
        # Streaming callback
        def stream_callback(chunk):
            """Nhận từng chunk và cập nhật UI"""
            self.current_answer_chunks.append(chunk)
            self.root.after(0, self._update_streaming_display)
        
        # Search in background
        def _search():
            try:
                self.is_loading = True
                self.search_btn.config(state=tk.DISABLED)
                self.status_label.config(text="🔍 Đang tìm kiếm tài liệu...", fg="#ffc107")
                self.root.update()
                
                temperature = self.temperature_var.get()
                top_k = self.topk_var.get()
                
                # Hiển thị đang xử lý
                self.answer_output.config(state=tk.NORMAL)
                self.answer_output.delete(1.0, tk.END)
                self.answer_output.insert(1.0, "⏳ Đang tìm kiếm và tạo câu trả lời...")
                self.answer_output.config(state=tk.DISABLED)
                
                # Rebuild chain
                qa_chain = build_rag_chain(temperature=temperature, top_k=top_k)
                
                # Query với streaming
                result = query_rag(qa_chain, question, stream_callback=stream_callback)
                
                # Lấy timing
                timing = result.get("timing", {})
                timing_text = f"\n\n⏱️ THỜI GIAN:\n   - Tìm kiếm: {timing.get('search_time', 0):.2f}s\n   - Tạo câu trả lời: {timing.get('llm_time', 0):.2f}s\n   - Tổng: {timing.get('total_time', 0):.2f}s"
                
                # Display final answer
                answer_text = result.get('answer', 'Không có câu trả lời')
                self.answer_output.config(state=tk.NORMAL)
                self.answer_output.delete(1.0, tk.END)
                self.answer_output.insert(1.0, answer_text + timing_text)
                self.answer_output.config(state=tk.DISABLED)
                
                # Display sources
                sources_text = ""
                for i, doc in enumerate(result.get("sources", []), 1):
                    content = doc.page_content[:150].replace('\n', ' ')
                    citation = doc.metadata.get("citation", "N/A")
                    sources_text += f"[{i}] {citation}\n{content}...\n\n"
                
                self.sources_output.config(state=tk.NORMAL)
                self.sources_output.delete(1.0, tk.END)
                self.sources_output.insert(1.0, sources_text)
                self.sources_output.config(state=tk.DISABLED)
                
                # Status với timing
                total_time = timing.get('total_time', 0)
                self.status_label.config(
                    text=f"✅ Hoàn thành {total_time:.2f}s | {len(result.get('sources', []))} tài liệu",
                    fg="#60d66d"
                )
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi tìm kiếm:\n\n{str(e)}")
                self.status_label.config(text="❌ Lỗi", fg="#d9534f")
            
            finally:
                self.is_loading = False
                self.search_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=_search, daemon=True)
        thread.start()
    
    def _update_streaming_display(self):
        """Cập nhật UI khi có chunk mới"""
        if self.current_answer_chunks:
            current_text = "".join(self.current_answer_chunks)
            self.answer_output.config(state=tk.NORMAL)
            self.answer_output.delete(1.0, tk.END)
            self.answer_output.insert(1.0, current_text + " ▌")
            self.answer_output.config(state=tk.DISABLED)
    
    def clear(self):
        """Clear inputs and outputs"""
        self.question_input.delete(1.0, tk.END)
        self.question_input.insert(1.0, "Ví dụ: Luật Thủy lợi quy định gì về bảo vệ công trình nước?")
        
        self.answer_output.config(state=tk.NORMAL)
        self.answer_output.delete(1.0, tk.END)
        self.answer_output.config(state=tk.DISABLED)
        
        self.sources_output.config(state=tk.NORMAL)
        self.sources_output.delete(1.0, tk.END)
        self.sources_output.config(state=tk.DISABLED)
        
        self.status_label.config(text="✅ Sẵn sàng", fg="#60d66d")

# ============================================================================
# MAIN
# ============================================================================

def main():
    root = tk.Tk()
    app = LegalAssistantApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
