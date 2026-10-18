"""Ground Qwen3-0.6B answers in the retrieved university IT policies."""
import os
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_tool import search_policy, verified_service_desk_phone

# Constant for unknown service desk phone reply
UNKNOWN_PHONE = (
    "I don't know the exact University IT Service Desk phone number. "
    "It is not specified in the available policy documents."
)


def get_default_model_path():
    """Get model path from environment or fallback to local directory / HuggingFace repo."""
    env_model_path = os.environ.get("QWEN_MODEL_PATH")
    if env_model_path:
        return env_model_path

    # Resolve sibling model folder
    script_dir = Path(__file__).resolve().parent.parent
    local_model_dir = script_dir / "SCSE_26_GEX3" / "models" / "qwen3-0.6b"
    return str(local_model_dir) if local_model_dir.is_dir() else "Qwen/Qwen3-0.6B"


class RAGAssistant:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or get_default_model_path()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype="auto"
        )
        self.model.to(self.device)
        self.model.eval()

    def answer(self, question: str) -> str:
        # Retrieve policy evidence from RAG tool
        policy_evidence = search_policy.invoke({"question": question})

        system_prompt = (
            "You are a university IT support assistant. Answer using only the "
            "policy evidence below. If the requested fact is not stated, say "
            "that you do not know it. Never invent a phone number. Keep the "
            "answer concise.\n\nPolicy evidence:\n" + policy_evidence
        )

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Build model input with chat template
        tokenized_inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        # Inference
        with torch.inference_mode():
            generation_output = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        generated_tokens = generation_output[0, tokenized_inputs["input_ids"].shape[1]:]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Check if user is asking for service desk phone number
        is_asking_phone = bool(
            re.search(r"service desk", question, re.IGNORECASE)
            and re.search(r"phone|telephone|number|call", question, re.IGNORECASE)
        )

        if is_asking_phone and verified_service_desk_phone() is None:
            return UNKNOWN_PHONE

        return raw_response
