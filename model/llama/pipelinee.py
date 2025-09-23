import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch

from conversation import ConversationManager
from rag import RAGRetriever
from lora import LoraTrainer
from model import MusicChatbot
from eval import ReRankerEvaluator


@dataclass
class PipelineConfig:
    # Paths
    songs_db_path: str = "scripts/crawl/songs.json"
    eval_data_path: str = "scripts/crawl/songs.json"
    train_data_path: str = "scripts/crawl/songs.json"
    lora_output_dir: str = "adapters/music_assistant"
    conversation_dir: str = "conversations"

    # Model
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_rag: bool = True
    use_lora: bool = True
    use_4bit: bool = True

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 5
    top_k_final: int = 3

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    train_epochs: int = 3
    train_batch_size: int = 1
    learning_rate: float = 2e-4

    # Conversation
    max_history_turns: int = 10
    context_turns: int = 3

    verbose: bool = True

    def __post_init__(self):
        # Auto create dirs
        for p in [self.songs_db_path, self.eval_data_path, self.train_data_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(self.lora_output_dir).parent.mkdir(parents=True, exist_ok=True)
        Path(self.conversation_dir).mkdir(parents=True, exist_ok=True)

    def validate(self):
        errors = []
        if self.use_rag and not Path(self.songs_db_path).exists():
            errors.append(f"❌ Songs DB not found: {self.songs_db_path}")
        if self.max_new_tokens <= 0:
            errors.append("❌ max_new_tokens phải > 0")
        if not (0 <= self.temperature <= 2):
            errors.append("❌ temperature phải trong khoảng [0, 2]")
        if self.top_k_final > self.top_k_retrieval:
            errors.append("❌ top_k_final không thể > top_k_retrieval")

        if errors:
            for e in errors: print(e)
            return False
        if self.verbose: print("✅ Config validation passed")
        return True

    def save(self, path="pipeline_config.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"✅ Config saved to {path}")

    @classmethod
    def load(cls, path="pipeline_config.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls(**json.load(f))
        except FileNotFoundError:
            print(f"⚠️ Config file {path} not found. Using default config.")
            return cls()
        except Exception as e:
            print(f"❌ Error loading config: {e}. Using default config.")
            return cls()


class MusicChatbotPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        if not self.config.validate():
            raise ValueError("❌ Invalid configuration")

        self.stats = {"total_queries": 0, "total_response_time": 0.0, "conversation_turns": 0}
        self._init_components()

    def _init_components(self):
        if self.config.verbose:
            print("🚀 Initializing components...")

        self.conversation_manager = ConversationManager(
            rag_db_path=self.config.songs_db_path,
            use_rag=self.config.use_rag,
            use_lora=self.config.use_lora
        )
        self.model = MusicChatbot(
            model_name=self.config.model_name,
            use_rag=self.config.use_rag
        )
        self.retriever = RAGRetriever(
            db_path=self.config.songs_db_path,
            embed_model=self.config.embedding_model
        ) if self.config.use_rag else None
        self.reranker = ReRankerEvaluator(
            eval_data_path=self.config.eval_data_path,
            rag_db_path=self.config.songs_db_path
        ) if Path(self.config.eval_data_path).exists() else None
        self.lora_trainer = LoraTrainer(
            base_model=self.config.model_name,
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            use_4bit=self.config.use_4bit
        ) if self.config.use_lora else None

        if self.config.verbose:
            print("✅ Components ready!")

    def chat(self, query: str, **override_params):
        start = time.time()
        gen_params = {
            "max_new_tokens": override_params.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": override_params.get("temperature", self.config.temperature),
            "top_p": override_params.get("top_p", self.config.top_p),
        }
        try:
            response = self.conversation_manager.generate_response(query, **gen_params)
            rt = time.time() - start
            self._update_stats(rt)
            return {
                "response": response,
                "response_time": round(rt, 3),
                "query": query,
                "config_used": {**gen_params, "model": self.config.model_name, "rag": self.config.use_rag}
            }
        except Exception as e:
            return {"response": f"Error: {e}", "error": str(e)}

    def evaluate(self, eval_queries=None):
        if not self.reranker:
            print("⚠️ No evaluator available")
            return None
        return self.reranker.run(eval_queries)

    def train_lora(self):
        if not self.lora_trainer:
            print("⚠️ LoRA not enabled")
            return False
        try:
            self.lora_trainer.train(
                dataset_path=self.config.train_data_path,
                output_dir=self.config.lora_output_dir,
                epochs=self.config.train_epochs,
                batch_size=self.config.train_batch_size,
                lr=self.config.learning_rate,
            )
            print("✅ LoRA training done!")
            return True
        except Exception as e:
            print(f"❌ LoRA training failed: {e}")
            return False

    def _update_stats(self, response_time: float):
        self.stats["total_queries"] += 1
        self.stats["total_response_time"] += response_time
        self.stats["conversation_turns"] = len(self.conversation_manager.conversation_history) // 2

    def get_stats(self):
        avg = self.stats["total_response_time"] / self.stats["total_queries"] if self.stats["total_queries"] else 0
        return {"total_queries": self.stats["total_queries"], "avg_time": round(avg, 3), "turns": self.stats["conversation_turns"]}

    def save_conversation(self, filename=None):
        ts = int(time.time())
        filename = filename or f"conversation_{ts}.json"
        path = Path(self.config.conversation_dir) / filename
        data = {"timestamp": ts, "config": asdict(self.config), "stats": self.get_stats(),
                "history": self.conversation_manager.conversation_history}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Conversation saved: {path}")
        return str(path)
