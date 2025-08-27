import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class MusicChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def build_prompt(self, user_input: str, context: str = None) -> str:
        system_prompt = "Bạn là chatbot âm nhạc, nhiệm vụ là trả lời dựa trên thông tin nhạc."
        if context:
            return f"{system_prompt}\n\nContext: {context}\n\nUser: {user_input}\nAssistant:"
        else:
            return f"{system_prompt}\n\nUser: {user_input}\nAssistant:"

    def generate_response(self, user_input: str, context: str = None,
                          max_new_tokens: int = 256, temperature: float = 0.7) -> str:

        prompt = self.build_prompt(user_input, context)
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        return outputs[0]['generated_text'].replace(prompt, "").strip()
