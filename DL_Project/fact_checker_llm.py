"""
Fact Checker LLM
Uses Llama 3 8B in 4-bit mode to verify claims based on RAG context.
"""

import torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline, AutoModel,
)

# -------------------------------------------------------
# LOGIN TO HUGGING FACE
# -------------------------------------------------------
# Replace YOUR_TOKEN_HERE with your real HF token.
login("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")


class FactCheckerLLM:
    """LLM that verifies news using retrieved evidence"""

    def __init__(self):
        print("Loading Llama 3 8B Fact Checker...")

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        # 4-BIT QUANTIZATION CONFIG
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load quantized model with accelerate
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id

        # Build pipeline (IMPORTANT: no device argument)
        self.llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )

        self.llm.model.generation_config.do_sample = False
        gen_cfg = self.llm.model.generation_config
        gen_cfg.do_sample = False


        #print("Fact-checking model loaded!\n")

    def verify_news_batch(self, input_query, llm_context):
        prompts = []

        for input, context in zip(input_query, llm_context):
            messages = [
                {
                    "role": "system",
                    "content": "You are a strict fact-checker. Use only the provided evidence."
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{input}\n\n"
                        f"Evidence:\n{context}\n\n"
                        "Return exactly one line:\n"
                        "VERDICT: SUPPORTED\n"
                        "or\n"
                        "VERDICT: REFUTED\n"
                        "or\n"
                        "VERDICT: NOT ENOUGH INFO"
                    )
                }
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(prompt)

        outputs = self.llm(
            prompts,
            max_new_tokens=25,
            do_sample=False,
            batch_size=10,
            return_full_text=False,
        )

        return [out[0]["generated_text"].strip() for out in outputs]

    def _format_response(self, llm_response: str, query: str, context: str) -> str:
        """Pretty output formatting"""

        out = "=" * 70 + "\n"
        out += "FACT CHECK RESULT\n"
        out += "=" * 70 + "\n\n"
        out += f"Query: {query}\n\n"
        out += f"Model Verdict:\n{llm_response}\n\n"
        out += "Evidence Used:\n"
        out += "-" * 70 + "\n"
        out += (context[:500] + "...\n") if len(context) > 500 else context + "\n"
        out += "=" * 70 + "\n"
        return llm_response
