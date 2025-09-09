# conversation.py
from model.music_chatbot import MusicChatbot
from model.llama.rag import RAGRetriever

class ConversationManager:
    def __init__(self, rag_db_path: str, use_rag=True, use_lora=False):
        self.chatbot = MusicChatbot(use_rag=use_rag)
        if use_rag:
            self.rag = RAGRetriever(rag_db_path)
        
        # Lưu lịch sử hội thoại để tạo ngữ cảnh
        self.conversation_history = []
        
        # Cờ để kiểm soát việc fine-tune với LoRA
        self.use_lora = use_lora
        if use_lora:
            self.load_lora_adapter()  # Hàm giả định để load LoRA weights

    def load_lora_adapter(self, adapter_path: str = "adapters/music_assistant"):
        # Code để load weights LoRA vào model (tùy thuộc vào thư viện bạn dùng)
        print(f"Loading LoRA adapter from {adapter_path}...")
        # Ví dụ: self.chatbot.model.load_adapter(adapter_path)

    def generate_response(self, user_input: str):
        # 1. Thu thập ngữ cảnh từ RAG
        context = ""
        if hasattr(self, 'rag'):
            retrieved_results = self.rag.retrieve(user_input, top_k=3)
            context = self._format_rag_results(retrieved_results)
            
        # 2. Thêm ngữ cảnh hội thoại gần đây (ví dụ: 3 tin nhắn gần nhất)
        recent_history = self._get_recent_history()
        
        # 3. Tạo prompt cuối cùng kết hợp mọi thứ
        final_context = recent_history + "\n" + context if context else recent_history
        
        # 4. Gọi model để generate response
        bot_response = self.chatbot.generate_response(
            user_input=user_input,
            context=final_context,
            max_new_tokens=512,
            temperature=0.7
        )
        
        # 5. Cập nhật lịch sử hội thoại
        self._update_conversation_history(user_input, bot_response)
        
        return bot_response

    def _format_rag_results(self, results):
        """Định dạng kết quả từ RAG thành text tự nhiên cho prompt."""
        formatted = "Thông tin bài hát liên quan:\n"
        for i, res in enumerate(results, 1):
            song = res['song_info']
            formatted += f"{i}. '{song['title']}' - {song['artist']} ({song['genre']})\n"
        return formatted

    def _get_recent_history(self, turn_count=3):
        """Lấy lịch sử hội thoại gần đây."""
        # Chỉ lấy số lượng turn nhất định
        recent_turns = self.conversation_history[-(turn_count*2):] if turn_count > 0 else []
        return "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" 
                         for i, msg in enumerate(recent_turns)])

    def _update_conversation_history(self, user_input, bot_response):
        """Cập nhật lịch sử hội thoại."""
        self.conversation_history.append(user_input)
        self.conversation_history.append(bot_response)
        # Giới hạn lịch sử để tránh tràn bộ nhớ
        if len(self.conversation_history) > 20:  # Giữ 10 cặp Q&A
            self.conversation_history = self.conversation_history[-20:]