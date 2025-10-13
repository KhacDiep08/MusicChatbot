import json
from typing import List, Dict, Any, Optional

class ConversationManager:
    def __init__(self, rag_db_path: str, use_rag=True, use_lora=True, top_k=3, chatbot_instance=None):
        from model import MusicChatbot
        from rag import RAGRetriever
        
        self.chatbot = chatbot_instance or MusicChatbot(use_rag=use_rag, use_int4=True)
        self.rag = RAGRetriever(rag_db_path) if use_rag else None
        self.conversation_history = []
        self.use_lora = use_lora
        self.top_k = top_k
        self.original_rag = self.rag

        if use_lora:
            self.load_lora_adapter()
    
    def set_rag_enabled(self, enabled: bool):
        self.rag = self.original_rag if enabled else None
            
    def load_lora_adapter(self, adapter_path: str = "adapters/music_assistant"):
        try:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.chatbot.model = PeftModel.from_pretrained(self.chatbot.model, adapter_path)
            print(" LoRA adapter loaded successfully")
        except Exception as e:
            print(f" Failed to load LoRA adapter: {e}")
            self.use_lora = False

    def generate_response(self, user_input: str, **gen_params):
        context = ""
        if self.rag:
            retrieved_results = self.rag.retrieve(user_input, top_k=self.top_k)
            context = self._format_rag_results(retrieved_results)

        recent_history = self._get_recent_history()
        
        if recent_history and context:
            final_context = f"{recent_history}\n\n{context}"
        elif context:
            final_context = context
        else:
            final_context = recent_history

        bot_response = self.chatbot.generate_response(
            user_input=user_input,
            context=final_context if final_context else None,
            **gen_params
        )

        self._update_conversation_history(user_input, bot_response)
        return bot_response

    def _format_rag_results(self, results: List[Dict], use_tags: bool = True) -> str:
        if not results:
            return ""
        
        try:
            if use_tags:
                formatted = "<database>\n"
                formatted += f"<total_songs>{len(results)}</total_songs>\n"
                
                for i, res in enumerate(results, 1):
                    doc_text = res["doc"]
                    score = res.get("score", 0.0)
                    
                    try:
                        parts = doc_text.split(" | ")
                        if len(parts) >= 1:
                            title_artist = parts[0].split(" - ", 1)
                            title = title_artist[0].strip() if len(title_artist) > 0 else "Unknown"
                            artist = title_artist[1].strip() if len(title_artist) > 1 else "Unknown"
                        else:
                            title = "Unknown"
                            artist = "Unknown"
                        
                        genre = parts[1].strip() if len(parts) > 1 else "Unknown"
                        lyrics = parts[2].strip() if len(parts) > 2 else ""
                        lyrics_preview = lyrics[:400] + "..." if len(lyrics) > 400 else lyrics
                        
                        formatted += f"\n<song id='{i}' relevance='{score:.3f}'>\n"
                        formatted += f"  <title>{self._escape_xml(title)}</title>\n"
                        formatted += f"  <artist>{self._escape_xml(artist)}</artist>\n"
                        formatted += f"  <genre>{self._escape_xml(genre)}</genre>\n"
                        if lyrics_preview:
                            formatted += f"  <lyrics>\n{self._escape_xml(lyrics_preview)}\n  </lyrics>\n"
                        formatted += "</song>\n"
                    
                    except Exception:
                        formatted += f"<song id='{i}' error='parse_failed'>\n"
                        formatted += f"  <raw>{self._escape_xml(doc_text[:200])}...</raw>\n"
                        formatted += "</song>\n"
                
                formatted += "</database>"
            else:
                formatted = "=== DỮ LIỆU CƠ SỞ ===\n"
                formatted += f"Tìm thấy {len(results)} bài:\n"
                
                for i, res in enumerate(results, 1):
                    doc_text = res["doc"]
                    parts = doc_text.split(" | ", 2)
                    if len(parts) >= 2:
                        title_artist = parts[0].split(" - ", 1)
                        title = title_artist[0] if len(title_artist) > 0 else "Unknown"
                        artist = title_artist[1] if len(title_artist) > 1 else "Unknown"
                        genre = parts[1] if len(parts) > 1 else "Unknown"
                        lyrics = parts[2] if len(parts) > 2 else ""
                        lyrics_preview = lyrics[:400] + "..." if len(lyrics) > 400 else lyrics
                        
                        formatted += f"\n[{i}] '{title}' - {artist} ({genre})\n"
                        if lyrics_preview:
                            formatted += f"Lời: {lyrics_preview}\n"
                    else:
                        formatted += f"{i}. {doc_text[:100]}...\n"
                
                formatted += "=== HẾT ==="
            
            return formatted.strip()
        
        except Exception as e:
            print(f"Error formatting RAG results: {e}")
            return ""

    def _escape_xml(self, text: str) -> str:
        if not text:
            return ""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))

    def _get_recent_history(self, turn_count: int = 3) -> str:
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
        self.conversation_history.extend([user_input, bot_response])
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self) -> List[str]:
        return self.conversation_history.copy()
