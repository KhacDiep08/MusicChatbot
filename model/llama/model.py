import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

class MusicChatbot:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = None, use_rag=True, use_int4=True):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        self.use_int4 = use_int4
        print(f"Loading model '{model_name}' on device '{self.device}' {'with INT4' if use_int4 else ''}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if use_int4 and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
            )

        model_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            print("🔄 Attempting to load model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            print("✅ Model loaded successfully with quantization")
        except ValueError as e:
            if "dispatched on the CPU or the disk" in str(e):
                print("⚠️ GPU memory insufficient, trying balanced device map...")
                model_kwargs["device_map"] = "balanced_low_0"
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    print("✅ Model loaded with balanced_low_0 device map")
                except Exception:
                    print("⚠️ Trying sequential device map...")
                    model_kwargs["device_map"] = "sequential"
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        print("✅ Model loaded with sequential device map")
                    except Exception:
                        print("⚠️ Fallback to FP16 without quantization...")
                        model_kwargs.pop("quantization_config", None)
                        model_kwargs["device_map"] = "auto"
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        self.use_int4 = False
                        print("✅ Model loaded in FP16 mode")
            else:
                raise e

        self.model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.device == "cuda" else None,
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1e9
            reserved = torch.cuda.memory_reserved()/1e9
            print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    def build_prompt(self, user_input: str, context: str = None, max_songs: int = 5) -> str:
        system_prompt = f"Trợ lý âm nhạc. Chỉ dùng dữ liệu được cung cấp. Gợi ý tối đa {max_songs} bài. Không bịa tên bài/nghệ sĩ."
        if context:
            context = context[:800] + "..." if len(context) > 800 else context
            user_message = f"DB: {context}\nQ: {user_input}\nA:"
        else:
            user_message = f"Q: {user_input}\nA: Cần dữ liệu để trả lời chính xác."
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

    def generate_response(self, user_input: str, context: str = None,
                          max_new_tokens=512, temperature=0.7, top_p=0.9) -> str:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prompt = self.build_prompt(user_input, context)
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"]
            return text.replace(prompt, "").strip()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
