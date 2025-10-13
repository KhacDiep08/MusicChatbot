import gradio as gr
import json
import time
from pipelinee import MusicChatbotPipeline, PipelineConfig

class MusicChatbotGUI:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.pipeline = MusicChatbotPipeline(self.config)
                
    def chat_response(self, message, history, use_rag, temperature, max_tokens):
        if not message.strip():
            return history, history, " Vui lòng nhập câu hỏi!"
        
        start_time = time.time()
        try:
            original_rag_state = self.pipeline.conversation_manager.rag
            if not use_rag:
                self.pipeline.conversation_manager.rag = None
                
            try:
                response = self.pipeline.conversation_manager.generate_response(
                    message, 
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                response_time = time.time() - start_time
                history.append([message, response])
                status = f" Đã trả lời trong {response_time:.2f}s | RAG: {'Bật' if use_rag else 'Tắt'}"
                return history, history, status
            finally:
                self.pipeline.conversation_manager.rag = original_rag_state
        except Exception as e:
            error_msg = f" Lỗi: {str(e)}"
            history.append([message, f"Xin lỗi, đã xảy ra lỗi: {str(e)}"])
            return history, history, error_msg

    def clear_conversation(self):
        self.pipeline.conversation_manager.clear_history()
        return [], [], " Đã xóa lịch sử hội thoại"

    def get_stats(self):
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
        if not history:
            return None
        filename = self.pipeline.save_conversation()
        return filename

    def create_interface(self):
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
                    chatbot = gr.Chatbot(height=500, label=" Cuộc trò chuyện", bubble_full_width=False)
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Hỏi gì đó về âm nhạc... (VD: 'Ed Sheeran có những bài hát nào?')",
                            label="Tin nhắn", lines=2, scale=4
                        )
                        send_btn = gr.Button(" Gửi", variant="primary", scale=1)
                    
                    with gr.Row():
                        gr.Examples(
                            examples=[
                                ["Tìm bài hát về tình yêu"],
                                ["Ed Sheeran có những bài hát nào?"],
                                ["Thể loại pop có gì hay?"],
                                ["Gợi ý bài hát buồn"],
                            ],
                            inputs=msg,
                            label="💡 Câu hỏi mẫu"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Cài đặt")
                    use_rag = gr.Checkbox(label="🔍 Bật RAG (Retrieval)", value=True)
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="🌡️ Temperature")
                    max_tokens = gr.Slider(minimum=128, maximum=1024, value=512, step=128, label="📝 Max Tokens")
                    
                    gr.Markdown("### 🛠️ Hành động")
                    clear_btn = gr.Button(" Xóa lịch sử", variant="secondary")
                    export_btn = gr.Button(" Xuất cuộc trò chuyện")
                    
                    gr.Markdown("###  Thống kê")
                    stats_display = gr.Markdown(self.get_stats())
                    refresh_stats_btn = gr.Button(" Cập nhật thống kê")
            
            status = gr.Textbox(label="Trạng thái", interactive=False, value=" Sẵn sàng chat!")
            history_state = gr.State([])
            
            def respond(message, history, use_rag, temperature, max_tokens):
                return self.chat_response(message, history, use_rag, temperature, max_tokens)
            
            send_btn.click(respond, inputs=[msg, history_state, use_rag, temperature, max_tokens],
                           outputs=[chatbot, history_state, status]).then(lambda: "", outputs=[msg])
            
            msg.submit(respond, inputs=[msg, history_state, use_rag, temperature, max_tokens],
                       outputs=[chatbot, history_state, status]).then(lambda: "", outputs=[msg])
            
            clear_btn.click(lambda: self.clear_conversation(),
                            outputs=[chatbot, history_state, status])
            
            export_btn.click(self.export_conversation, inputs=[history_state],
                             outputs=[gr.File(label="📁 File đã xuất")])
            
            refresh_stats_btn.click(self.get_stats, outputs=[stats_display])
            
        return demo

def main():
    config = PipelineConfig.load("config.json")
    gui = MusicChatbotGUI(config)
    demo = gui.create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, show_error=True, debug=True)

if __name__ == "__main__":
    main()
