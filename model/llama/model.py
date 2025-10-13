import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)


class MusicChatbot:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = None,
        use_rag=True,
        use_int4=True,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        self.use_int4 = use_int4

        print(f"Loading model '{model_name}' on device '{self.device}' {'with INT4' if use_int4 else ''}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cấu hình lượng tử hóa (quantization)
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

        # Load model
        try:
            print("Attempting to load model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            print("Model loaded successfully with quantization.")
        except ValueError as e:
            if "dispatched on the CPU or the disk" in str(e):
                print("GPU memory insufficient, trying balanced device map...")
                model_kwargs["device_map"] = "balanced_low_0"

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    print("Model loaded with balanced_low_0 device map.")
                except Exception:
                    print("Trying sequential device map...")
                    model_kwargs["device_map"] = "sequential"

                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        print("Model loaded with sequential device map.")
                    except Exception:
                        print("Fallback to FP16 without quantization...")
                        model_kwargs.pop("quantization_config", None)
                        model_kwargs["device_map"] = "auto"
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        self.use_int4 = False
                        print("Model loaded in FP16 mode.")
            else:
                raise e

        self.model.eval()

        # Pipeline sinh văn bản
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.device == "cuda" else None,
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    def build_prompt(self, user_input: str, context: str = None, max_songs: int = 5) -> str:
        """
        Xây dựng prompt đầu vào cho mô hình, sử dụng few-shot learning.
        """
        system_prompt = f"Bạn là một Trợ lý Âm nhạc. Hãy làm theo các ví dụ một cách cẩn thận. Gợi ý tối đa {max_songs} bài."

        examples = """
### Ví dụ 1: Tìm kiếm thành công ###
[INST] ### Dữ liệu Cung cấp ###
- Taylor Swift - Love Story
- Taylor Swift - Blank Space
- Lady Gaga - Bad Romance
### Yêu cầu của người dùng ###
Gợi ý vài bài hát của Taylor Swift
A: [/INST]
- Love Story
- Blank Space
</s>
<s>[INST] ### Ví dụ 2: Không tìm thấy nghệ sĩ ###
### Dữ liệu Cung cấp ###
- Taylor Swift - Love Story
- Taylor Swift - Blank Space
### Yêu cầu của người dùng ###
Tìm giúp tôi bài hát của Adele
A: [/INST]
Xin lỗi, trong dữ liệu được cung cấp, tôi không tìm thấy bài hát nào của nghệ sĩ Adele.
</s>
<s>[INST] ### Ví dụ 3: Không có dữ liệu ###
### Dữ liệu Cung cấp ###
(Không có dữ liệu nào được cung cấp)
### Yêu cầu của người dùng ###
Gợi ý cho tôi vài bài nhạc pop
A: [/INST]
Tôi cần dữ liệu về các bài hát để có thể đưa ra gợi ý chính xác.
"""

        # Dữ liệu đầu vào hiện tại
        if context:
            context = context[:800] + "..." if len(context) > 800 else context
            current_request = f"""### Dữ liệu Cung cấp ###
{context}

### Yêu cầu của người dùng ###
{user_input}

A: """
        else:
            current_request = f"""### Dữ liệu Cung cấp ###
(Không có dữ liệu nào được cung cấp)

### Yêu cầu của người dùng ###
{user_input}

A: """

        return (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{examples.strip()}\n</s>\n<s>[INST] {current_request.strip()} [/INST]"
        )

    def generate_response(
        self,
        user_input: str,
        context: str = None,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    ) -> str:
        """
        Sinh phản hồi từ mô hình dựa trên prompt đã xây dựng.
        """
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
