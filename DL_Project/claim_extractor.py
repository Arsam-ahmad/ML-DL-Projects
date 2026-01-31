"""
Verifiable Claim Extractor
Uses Llama 3 8B in 4-bit mode to extract factual claims from news text.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional

import torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# LOGIN
login("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")  # same token as fact checker


class ClaimExtractor:
    """Extracts factual verifiable statements using Llama 3"""

    def __init__(self):
        print("Loading Llama 3 8B Claim Extractor...")

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        print("Claim extractor loaded!\n")

    # ------------------------------------------------------------------------
    # Guardian Article Helpers
    # ------------------------------------------------------------------------

    def extract_from_guardian_article(self, article_data: Dict) -> List[Dict[str, str]]:
        """Extract claims from Guardian API article data"""

        article_text = self._get_article_text(article_data)
        article_date = self._get_article_date(article_data)

        if not article_text:
            return []

        claims = self._extract_claims_from_text(article_text)

        return [
            {
                "claim": c,
                "date": article_date,
                "formatted": f"{c} | {article_date}",
            }
            for c in claims
        ]

    def _get_article_text(self, article_data: Dict) -> Optional[str]:
        """Handle various Guardian response formats"""
        try:
            if "fields" in article_data and "bodyText" in article_data["fields"]:
                return article_data["fields"]["bodyText"]
            if "bodyText" in article_data:
                return article_data["bodyText"]
            if "body" in article_data:
                return article_data["body"]
            return None
        except Exception:
            return None

    def _get_article_date(self, article_data: Dict) -> str:
        """Extract webPublicationDate → YYYY-MM-DD"""
        try:
            d = article_data.get("webPublicationDate", "")
            if d:
                return datetime.fromisoformat(d.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            return datetime.now().strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------------
    # Claim Extraction Logic
    # ------------------------------------------------------------------------

    def _extract_claims_from_text(self, text: str) -> List[str]:
        """Splits long text and extracts claims chunk-by-chunk."""

        chunks = self._split_text_into_chunks(text, max_length=350)

        claims = []
        for chunk in chunks:
            claims.extend(self._extract_claims_from_chunk(chunk))

        # Remove duplicates
        return list(dict.fromkeys(claims))

    def _split_text_into_chunks(self, text: str, max_length: int = 350) -> List[str]:
        """Split into sentence blocks for LLM processing"""
        sentences = re.split(r"[.!?]+\s+", text)
        chunks, cur, cnt = [], [], 0

        for s in sentences:
            if not s.strip():
                continue
            words = len(s.split())
            if cnt + words > max_length and cur:
                chunks.append(" ".join(cur))
                cur = [s]
                cnt = words
            else:
                cur.append(s)
                cnt += words

        if cur:
            chunks.append(" ".join(cur))

        return chunks

    def _extract_claims_from_chunk(self, text: str) -> List[str]:
        """LLM call to extract claims"""

        prompt = f"""
Extract ONLY factual, verifiable claims from the text below.

A verifiable claim must be something that can be confirmed with evidence.
Do NOT return opinions, predictions, or vague statements.
Return one claim per line.

Text:
{text}

Claims:
"""

        outputs = self.generator(
            prompt,
            max_new_tokens=256,
            do_sample=False,
        )

        generated = outputs[0]["generated_text"]

        # Clean prompt echo
        if generated.startswith(prompt):
            result = generated[len(prompt):].strip()
        else:
            result = generated.strip()

        return self._parse_claims_from_response(result)

    def _parse_claims_from_response(self, resp: str) -> List[str]:
        """Clean up each LLM output line"""

        claims = []
        for line in resp.split("\n"):
            l = line.strip()
            l = re.sub(r"^[\-\d\.\*]+\s*", "", l)  # remove leading bullets/numbers

            if len(l) < 10:
                continue
            if l.lower().startswith(("claims:", "here", "the following")):
                continue

            claims.append(l)

        return claims
