import json
from typing import List, Dict, Any, Optional

class ConversationManager:
    def __init__(self, rag_db_path: str, use_rag=True, use_lora=False, top_k=3):
        # Delay imports để tránh circular imports
        from model import MusicChatbot
        from rag import RAGRetriever
        
        self.chatbot = MusicChatbot(use_rag=use_rag)
        self.rag = RAGRetriever(rag_db_path) if use_rag else None
        self.conversation_history = []
        self.use_lora = use_lora
        self.top_k = top_k
        
        if use_lora:
            self.load_lora_adapter()

    def load_lora_adapter(self, adapter_path: str = "adapters/music_assistant"):
        """Thực sự load LoRA adapter"""
        try:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.chatbot.model = PeftModel.from_pretrained(
                self.chatbot.model, 
                adapter_path
            )
            print("✅ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LoRA adapter: {e}")
            # Fallback to original model
            self.use_lora = False

    def generate_response(self, user_input: str, **gen_params):
        # Lấy context từ RAG nếu bật
        context = ""
        if self.rag:
            retrieved_results = self.rag.retrieve(user_input, top_k=self.top_k)
            context = self._format_rag_results(retrieved_results)

        # Lấy history gần nhất
        recent_history = self._get_recent_history()
        
        # Kết hợp context và history
        if recent_history and context:
            final_context = f"{recent_history}\n\n{context}"
        elif context:
            final_context = context
        else:
            final_context = recent_history

        # Sinh câu trả lời từ model
        bot_response = self.chatbot.generate_response(
            user_input=user_input,
            context=final_context if final_context else None,
            **gen_params
        )

        # Cập nhật history
        self._update_conversation_history(user_input, bot_response)
        return bot_response

    def _format_rag_results(self, results: List[Dict]) -> str:
        """Format kết quả từ RAGRetriever"""
        if not results:
            return ""
        
        try:
            # Phân tích cú pháp doc string để lấy thông tin bài hát
            formatted = "Thông tin bài hát liên quan:\n"
            for i, res in enumerate(results, 1):
                doc_text = res["doc"]
                
                # Phân tích doc_text để trích xuất thông tin
                # Format: "title - artist | genre | lyrics"
                parts = doc_text.split(" | ", 2)
                if len(parts) >= 2:
                    title_artist = parts[0].split(" - ", 1)
                    title = title_artist[0] if len(title_artist) > 0 else "Unknown"
                    artist = title_artist[1] if len(title_artist) > 1 else "Unknown"
                    genre = parts[1] if len(parts) > 1 else "Unknown"
                    
                    formatted += f"{i}. '{title}' - {artist} ({genre})\n"
                else:
                    # Fallback: hiển thị toàn bộ doc text
                    formatted += f"{i}. {doc_text[:100]}...\n"
                    
            return formatted.strip()
            
        except Exception as e:
            print(f"Error formatting RAG results: {e}")
            return ""

    def _get_recent_history(self, turn_count: int = 3) -> str:
        """Lấy lịch sử hội thoại gần nhất"""
        if turn_count <= 0 or not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-(turn_count*2):]
        history_lines = []
        
        for i in range(0, len(recent_turns), 2):
            if i < len(recent_turns):
                history_lines.append(f"User: {recent_turns[i]}")
            if i + 1 < len(recent_turns):
                history_lines.append(f"Assistant: {recent_turns[i+1]}")
                
        return "\n".join(history_lines)

    def _update_conversation_history(self, user_input: str, bot_response: str):
        """Cập nhật lịch sử hội thoại"""
        self.conversation_history.extend([user_input, bot_response])
        
        # Giữ lại tối đa 20 message (10 cặp Q-A)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self):
        """Xóa lịch sử hội thoại"""
        self.conversation_history = []
    
    def get_history(self) -> List[str]:
        """Lấy toàn bộ lịch sử hội thoại"""
        return self.conversation_history.copy()