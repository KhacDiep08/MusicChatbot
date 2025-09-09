import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from model.llama.rag import RAGRetriever  # Giả sử bạn đã có module này

class MusicChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = None, use_rag=False):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        print(f"Loading model '{model_name}' on device '{self.device}'...")

        # if self.use_rag:
        #     self.rag = RAGRetriever("data/songs.json")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Sử dụng low_cpu_mem_usage và torch_dtype để tối ưu bộ nhớ
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def build_prompt(self, user_input: str, context: str = None) -> str:
        system_prompt = "Bạn là một trợ lý âm nhạc hữu ích và thân thiện. Hãy trả lời câu hỏi của người dùng dựa trên thông tin ngữ cảnh được cung cấp (nếu có)."
        
        if context:
            user_message_with_context = f"Dựa trên thông tin sau:\n{context}\n\nHãy trả lời câu hỏi: {user_input}"
        else:
            user_message_with_context = user_input

        # Định dạng prompt chuẩn cho Llama 2-Chat
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message_with_context} [/INST]"
        return prompt

    def generate_response_no_rag(self, user_input: str,
                                 context: str = None,
                                 max_new_tokens: int = 512,  # Tăng lên một chút
                                 temperature: float = 0.7,
                                 top_p: float = 0.9,
                                 repetition_penalty: float = 1.1) -> str:
        """
        Trả lời KHÔNG dùng RAG (chỉ dựa trên model).
        """
        prompt = self.build_prompt(user_input, context)
        
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,  # Giúp model biết điểm dừng
        )
        
        full_generated_text = outputs[0]['generated_text']
        # Chỉ lấy phần trả lời sau prompt
        assistant_response = full_generated_text.replace(prompt, "")
        return assistant_response.strip()

    def generate_response(self, user_input: str,
                          max_new_tokens=512,
                          temperature=0.7) -> str:
        """
        Trả lời có dùng RAG nếu bật.
        """
        context = ""
        if self.use_rag:
            # retrieved = self.rag.retrieve(user_input, top_k=3)
            # context = "\n".join([r["doc"] for r in retrieved])
            # Giả lập dữ liệu RAG cho ví dụ
            context = "Bài hát 'Shape of You' của Ed Sheeran là một bản hit pop nằm trong album '÷' (Divide) phát hành năm 2017."
            # TODO: Thêm logic kiểm tra nếu RAG không tìm thấy gì, có thể fallback một cách thông minh

        return self.generate_response_no_rag(
            user_input=user_input,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

# Example usage
if __name__ == "__main__":
    bot = MusicChatbot(use_rag=False)
    response = bot.generate_response("Bài hát 'Shape of You' là của ai?")
    print(response)