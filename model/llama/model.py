import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

class MusicChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = None, use_rag=True, use_int4=True):
        # Tự động chọn device nếu không set
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        self.use_int4 = use_int4
        print(f"Loading model '{model_name}' on device '{self.device}' {'with INT4' if use_int4 else ''}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # INT4 Quantization config
        quantization_config = None
        if use_int4 and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load model
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cuda":
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto"
            })
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def build_prompt(self, user_input: str, context: str = None, max_songs: int = 5) -> str:
        """Tạo prompt với validation"""
    
        system_prompt = f"""Trợ lý âm nhạc. Chỉ dùng dữ liệu được cung cấp. Gợi ý tối đa {max_songs} bài. Không bịa tên bài/nghệ sĩ."""
    
        if context:
            user_message = f"DB: {context[:1000]}...\nQ: {user_input}\nA:"  # Truncate context nếu quá dài
        else:
            user_message = f"Q: {user_input}\nA: Cần dữ liệu để trả lời chính xác."

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
