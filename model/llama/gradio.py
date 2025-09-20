import gradio as gr
import json
import time
from pipeline import MusicChatbotPipeline, PipelineConfig

class MusicChatbotGUI:
    def __init__(self, config: PipelineConfig = None):
        # Initialize pipeline với config
        self.config = config or PipelineConfig()
        self.pipeline = MusicChatbotPipeline(self.config)
        
        # Backup pipeline for comparison
        config_no_rag = PipelineConfig(**{**self.config.__dict__, 'use_rag': False})
        self.pipeline_no_rag = MusicChatbotPipeline(config_no_rag)
        
    def chat_response(self, message, history, use_rag, temperature, max_tokens):
        """Generate chatbot response with timing"""
        if not message.strip():
            return history, history, "⚠️ Vui lòng nhập câu hỏi!"
        
        start_time = time.time()
        
        try:
            # Choose conversation manager
            conv_manager = self.conv_with_rag if use_rag else self.conv_without_rag
            
            # Generate response
            response = conv_manager.generate_response(message)
            
            # Calculate timing
            response_time = time.time() - start_time
            self.total_queries += 1
            self.avg_response_time = (self.avg_response_time * (self.total_queries - 1) + response_time) / self.total_queries
            
            # Update history
            history.append([message, response])
            
            # Status message
            status = f"✅ Đã trả lời trong {response_time:.2f}s | RAG: {'Bật' if use_rag else 'Tắt'}"
            
            return history, history, status
            
        except Exception as e:
            error_msg = f"❌ Lỗi: {str(e)}"
            history.append([message, f"Xin lỗi, đã xảy ra lỗi: {str(e)}"])
            return history, history, error_msg

    def clear_conversation(self):
        """Clear conversation history"""
        self.pipeline.conversation_manager.conversation_history = []
        self.pipeline_no_rag.conversation_manager.conversation_history = []
        return [], [], "🔄 Đã xóa lịch sử hội thoại"

    def get_stats(self):
        """Get chatbot statistics"""
        stats = self.pipeline.get_stats()
        rag_size = len(self.pipeline.retriever.db) if self.pipeline.retriever else 0
        
        return f"""
        📊 **Thống kê Chat:**
        - Tổng số câu hỏi: {stats['total_queries']}
        - Thời gian trả lời TB: {stats['avg_time']}s
        - Số lượt hội thoại: {stats['turns']}
        - RAG database: {rag_size} bài hát
        - Model: {self.config.model_name.split('/')[-1]}
        - LoRA: {'Bật' if self.config.use_lora else 'Tắt'}
        """

    def export_conversation(self, history):
        """Export conversation to JSON"""
        if not history:
            return None
            
        # Use pipeline's save method
        filename = self.pipeline.save_conversation()
        return filename

    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-message {
            padding: 10px;
            margin: 5px;
            border-radius: 10px;
        }
        """
        
        with gr.Blocks(css=css, title="🎵 Music Chatbot", theme=gr.themes.Soft()) as demo:
            
            # Header
            gr.Markdown("""
            # 🎵 Music Chatbot Assistant
            ### Trợ lý AI chuyên về âm nhạc với công nghệ RAG + Llama-2
            
            💡 **Hướng dẫn sử dụng:**
            - Hỏi về bài hát, ca sĩ, thể loại nhạc
            - Bật RAG để tìm kiếm trong cơ sở dữ liệu
            - Điều chỉnh tham số để tùy chỉnh phản hồi
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main chatbot interface
                    chatbot = gr.Chatbot(
                        height=500,
                        label="🤖 Cuộc trò chuyện",
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Hỏi gì đó về âm nhạc... (VD: 'Ed Sheeran có những bài hát nào?')",
                            label="Tin nhắn",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("📤 Gửi", variant="primary", scale=1)
                    
                    # Quick examples
                    with gr.Row():
                        gr.Examples(
                            examples=[
                                ["Tìm bài hát về tình yêu"],
                                ["Ed Sheeran có những bài hát nào?"],
                                ["Thể loại pop có gì hay?"],
                                ["Gợi ý bài hát buồn"],
                                ["Nhạc sĩ Trịnh Công Sơn viết những gì?"]
                            ],
                            inputs=msg,
                            label="💡 Câu hỏi mẫu"
                        )
                
                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### ⚙️ Cài đặt")
                    
                    use_rag = gr.Checkbox(
                        label="🔍 Bật RAG (Retrieval)",
                        value=True,
                        info="Tìm kiếm trong cơ sở dữ liệu nhạc"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Độ sáng tạo (cao = ngẫu nhiên hơn)"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=128,
                        label="📝 Max Tokens",
                        info="Độ dài tối đa của phản hồi"
                    )
                    
                    # Action buttons
                    gr.Markdown("### 🛠️ Hành động")
                    clear_btn = gr.Button("🗑️ Xóa lịch sử", variant="secondary")
                    export_btn = gr.Button("💾 Xuất cuộc trò chuyện")
                    
                    # Stats
                    gr.Markdown("### 📊 Thống kê")
                    stats_display = gr.Markdown(self.get_stats())
                    refresh_stats_btn = gr.Button("🔄 Cập nhật thống kê")
            
            # Status bar
            status = gr.Textbox(
                label="Trạng thái",
                interactive=False,
                value="✅ Sẵn sàng chat!"
            )
            
            # Hidden state for history
            history_state = gr.State([])
            
            # Event handlers
            def respond(message, history, use_rag, temperature, max_tokens):
                return self.chat_response(message, history, use_rag, temperature, max_tokens)
            
            # Send message
            send_btn.click(
                respond,
                inputs=[msg, history_state, use_rag, temperature, max_tokens],
                outputs=[chatbot, history_state, status]
            ).then(
                lambda: "",  # Clear input
                outputs=[msg]
            )
            
            # Enter key
            msg.submit(
                respond,
                inputs=[msg, history_state, use_rag, temperature, max_tokens],
                outputs=[chatbot, history_state, status]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            # Clear conversation
            clear_btn.click(
                lambda: ([], [], "🔄 Đã xóa lịch sử hội thoại"),
                outputs=[chatbot, history_state, status]
            ).then(
                self.clear_conversation
            )
            
            # Export conversation
            export_btn.click(
                self.export_conversation,
                inputs=[history_state],
                outputs=[gr.File(label="📁 File đã xuất")]
            )
            
            # Refresh stats
            refresh_stats_btn.click(
                self.get_stats,
                outputs=[stats_display]
            )
            
        return demo

def main():
    # Load config hoặc dùng default
    config = PipelineConfig.load("config.json")  # Auto fallback to default nếu không có
    
    # Initialize GUI với config
    gui = MusicChatbotGUI(config)
    
    # Create and launch interface
    demo = gui.create_interface()
    
    # Launch with options
    demo.launch(
        share=True,  # Tạo public link
        server_name="0.0.0.0",  # Accept từ mọi IP
        server_port=7860,  # Port mặc định
        show_error=True,  # Hiện lỗi chi tiết
        debug=True  # Debug mode
    )

if __name__ == "__main__":
    main()