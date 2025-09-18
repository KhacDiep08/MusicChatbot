import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class MusicChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = None, use_rag=False):
        # Tự động chọn device nếu không set
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        print(f"Loading model '{model_name}' on device '{self.device}'...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        # Generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def build_prompt(self, user_input: str, context: str = None) -> str:
        """Tạo prompt chuẩn cho model"""
        system_prompt = "Bạn là một trợ lý âm nhạc hữu ích và thân thiện."
        if context:
            user_message = f"Dựa trên thông tin sau:\n{context}\n\nHãy trả lời: {user_input}"
        else:
            user_message = user_input

        # Prompt theo format của LLaMA
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

    def generate_response(self, user_input: str, context: str = None,
                          max_new_tokens=512, temperature=0.7, top_p=0.9) -> str:
        """Sinh câu trả lời"""
        prompt = self.build_prompt(user_input, context)
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = outputs[0]["generated_text"]
        return text.replace(prompt, "").strip()
