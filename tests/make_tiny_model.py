"""Build a tiny, randomly initialised Llama model + Llama-3-style tokenizer.

Used only by the smoke tests, so the whole training / evaluation pipeline can be
checked on a laptop CPU in under a minute without downloading 16 GB of weights.
Run:  python -m tests.make_tiny_model
"""
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src import config
from src.prepare_data import load_jsonl
from src.prompts import build_user_prompt, SYSTEM_PROMPT

OUT = config.MODELS_DIR / "tiny-llama-test"
SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def main(out: Path = OUT):
    rows = load_jsonl(config.TRAIN_PATH)[:500]
    corpus = [SYSTEM_PROMPT, build_user_prompt("x", "y", "z")] + [
        f"{r['supplier_name']} {r['line_description']} {r['department']} {r['category']}" for r in rows]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=2000, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>",
                                     eos_token="<|end_of_text|>", additional_special_tokens=SPECIAL[2:])
    hf_tok.chat_template = LLAMA3_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      bos_token_id=hf_tok.bos_token_id, eos_token_id=hf_tok.eos_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    hf_tok.save_pretrained(out)
    print(f"tiny model -> {out}")


if __name__ == "__main__":
    main()
