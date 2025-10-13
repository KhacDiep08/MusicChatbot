import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


class MusicChatbot:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", device=None, use_rag=True, use_int4=True):
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

    def build_prompt_english(self, user_input: str, context: str = None, max_songs: int = 5) -> str:
        system_prompt = f"""You are a music information retrieval system, NOT a creative conversational assistant.

**ULTIMATE RULES YOU MUST FOLLOW:**
1. **ABSOLUTELY ONLY** use information from the "### Provided Data ###" section.
2. **NEVER** use your internal, pre-existing knowledge.
3. **NEVER** invent song titles or artists that are not in the provided data.
4. If the provided data is insufficient to fulfill the request, provide what you can and explicitly state that the information is incomplete.
5. Suggest a maximum of {max_songs} songs.
6. Always follow the format and style of the provided examples.
"""

        examples = """
### Example 1: Successful Search ###
[INST] ### Provided Data ###
- Taylor Swift - Love Story
- Taylor Swift - Blank Space
- Lady Gaga - Bad Romance
### User Request ###
Suggest some songs by Taylor Swift
A: [/INST]
- Love Story
- Blank Space
</s>
<s>[INST] ### Example 2: Artist Not Found ###
### Provided Data ###
- Taylor Swift - Love Story
- Taylor Swift - Blank Space
### User Request ###
Find a song by Adele for me
A: [/INST]
I'm sorry, but I could not find any songs by the artist Adele in the provided data.
</s>
<s>[INST] ### Example 3: No Data Provided ###
### Provided Data ###
(No data provided)
### User Request ###
Suggest some pop songs for me
A: [/INST]
I need data about songs to provide an accurate suggestion.
"""

        if context:
            context = context[:800] + "..." if len(context) > 800 else context
            current_request = f"""### Provided Data ###
{context}

### User Request ###
{user_input}

A: """
        else:
            current_request = f"""### Provided Data ###
(No data provided)

### User Request ###
{user_input}

A: """

        return (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{examples.strip()}\n</s>\n<s>[INST] {current_request.strip()} [/INST]"
        )

    def generate_response(self, user_input: str, context: str = None, max_new_tokens=512, temperature=0.7, top_p=0.9) -> str:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prompt = self.build_prompt_english(user_input, context)
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
