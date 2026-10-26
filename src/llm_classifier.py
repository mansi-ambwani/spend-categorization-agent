"""Inference with the fine-tuned Llama-3 + LoRA adapter.

The model generates the category name; we then
  * snap the text to the closest valid category (guards against small spelling drift), and
  * compute a confidence score = probability the model assigned to the generated answer.
Low-confidence rows can be routed to a human for review.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config
from src.prompts import build_messages, category_names, snap_to_category


class LLMSpendClassifier:
    def __init__(self, adapter_dir: str | Path | None = config.ADAPTER_DIR, base_model: str | None = None,
                 load_in_4bit: bool = True):
        adapter_dir = Path(adapter_dir) if adapter_dir else None
        has_adapter = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
        if base_model is None:
            base_model = config.BASE_MODEL
            if has_adapter:  # use the base model the adapter was trained on
                import json
                base_model = json.loads((adapter_dir / "adapter_config.json").read_text()).get(
                    "base_model_name_or_path", base_model)
        self.categories = category_names()
        tok_src = str(adapter_dir) if has_adapter else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, token=config.HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation

        cuda = torch.cuda.is_available()
        kwargs = {"token": config.HF_TOKEN}
        if cuda and load_in_4bit:
            from transformers import BitsAndBytesConfig

            dtype = torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8) else torch.float16
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype), device_map={"": 0}, dtype=dtype)
        elif cuda:
            kwargs.update(device_map={"": 0}, dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if has_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        else:
            print("NOTE: no LoRA adapter found -> running the base model zero-shot.")
        self.model = model.eval()
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.stop_ids = [i for i in {self.tokenizer.eos_token_id, eot} if i is not None and i >= 0]

    @torch.no_grad()
    def predict(self, records: list[dict], batch_size: int = 16, progress: bool = True) -> list[dict]:
        results = []
        n_batches = (len(records) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = records[b * batch_size: (b + 1) * batch_size]
            prompts = [self.tokenizer.apply_chat_template(
                build_messages(r["supplier_name"], r["line_description"], r.get("department", "")),
                tokenize=False, add_generation_prompt=True) for r in batch]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model.generate(**enc, max_new_tokens=12, do_sample=False, num_beams=1,
                                      eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id,
                                      output_scores=True, return_dict_in_generate=True)
            gen = out.sequences[:, enc["input_ids"].shape[1]:]
            logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            for i in range(len(batch)):
                toks, lp = gen[i].tolist(), logp[i]
                n = len(toks)
                for j, t in enumerate(toks):  # stop at end-of-turn / padding
                    if t in self.stop_ids or t == self.tokenizer.pad_token_id:
                        n = j
                        break
                text = self.tokenizer.decode(toks[:n], skip_special_tokens=True)
                conf = float(torch.exp(lp[:n].float().sum()).item()) if n > 0 else 0.0
                results.append({"category": snap_to_category(text, self.categories),
                                "raw_output": text, "confidence": round(conf, 4)})
            if progress:
                print(f"\r  classified {min((b + 1) * batch_size, len(records))}/{len(records)}", end="")
        if progress:
            print()
        return results
