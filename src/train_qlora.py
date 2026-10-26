"""Fine-tune Llama-3 8B Instruct with QLoRA (4-bit base model + LoRA adapters via PEFT).

What happens:
  1. The 8B base model is loaded in 4-bit NF4 precision (bitsandbytes) -> fits in ~6 GB of GPU memory.
  2. Small trainable LoRA matrices are attached to every attention / MLP projection (PEFT).
     Only these (~0.5% of parameters) are trained; the 8B base weights stay frozen.
  3. Each training example is the chat prompt (taxonomy + line item) followed by the correct
     category.  The loss is computed ONLY on the answer tokens (prompt tokens are masked with -100).
  4. The adapter (~80 MB) is saved to models/llama3-spend-qlora/.

Needs an NVIDIA GPU with >= 12 GB memory (free Google Colab / Kaggle T4 works).
Run:  python -m src.train_qlora            (full run)
      python -m src.train_qlora --max-train-samples 1500   (quicker run)
"""
import argparse
import json
import math
import os
import random

# QLoRA trains on a single GPU. On machines with 2 GPUs (e.g. Kaggle T4 x2) hide the second one,
# otherwise the Trainer wraps the 4-bit model in DataParallel and crashes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src import config
from src.prepare_data import load_jsonl
from src.prompts import build_messages


def end_of_turn_id(tokenizer) -> int:
    """Llama-3 ends an assistant turn with <|eot_id|>; fall back to eos for other models."""
    tid = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if tid is None or tid == tokenizer.unk_token_id:
        tid = tokenizer.eos_token_id
    return tid


def prompt_text(tokenizer, rec: dict) -> str:
    msgs = build_messages(rec["supplier_name"], rec["line_description"], rec.get("department", ""))
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class SFTDataset(Dataset):
    """Tokenised (prompt + answer) pairs with the prompt part masked out of the loss."""

    def __init__(self, records, tokenizer, max_len):
        eot = end_of_turn_id(tokenizer)
        self.items = []
        for r in records:
            p_ids = tokenizer(prompt_text(tokenizer, r), add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(r["category"], add_special_tokens=False)["input_ids"] + [eot]
            ids = (p_ids + a_ids)[-max_len:]
            labels = ([-100] * len(p_ids) + a_ids)[-max_len:]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * (L - n))
            labels.append(b["labels"] + [-100] * (L - n))
            mask.append([1] * n + [0] * (L - n))
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def load_base_model(base_model: str, use_4bit: bool):
    kwargs = {"token": config.HF_TOKEN}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = compute_dtype
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=config.BASE_MODEL)
    ap.add_argument("--output-dir", default=str(config.ADAPTER_DIR))
    ap.add_argument("--epochs", type=float, default=config.NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM_STEPS)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1, help="override number of steps (for smoke tests)")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit quantisation (CPU smoke test)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and not args.no_4bit
    if not use_cuda:
        print("WARNING: no CUDA GPU found - running on CPU (only sensible with a tiny test model).")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=config.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train = load_jsonl(config.TRAIN_PATH)
    val = load_jsonl(config.VAL_PATH)
    random.shuffle(train)
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    val = val[:200]  # keep periodic evaluation fast
    train_ds = SFTDataset(train, tokenizer, config.MAX_SEQ_LEN)
    val_ds = SFTDataset(val, tokenizer, config.MAX_SEQ_LEN)
    lens = [len(x["input_ids"]) for x in train_ds.items]
    print(f"train={len(train_ds)} val={len(val_ds)}  tokens/example: mean={sum(lens)/len(lens):.0f} max={max(lens)}")

    model = load_base_model(args.base_model, use_4bit)
    model.config.use_cache = False
    gc_kwargs = {"use_reentrant": False}
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=gc_kwargs)
    lora = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                      target_modules=config.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    bf16 = use_cuda and (torch.cuda.get_device_capability()[0] >= 8)
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    eval_every = max(10, steps_per_epoch // 4)
    targs = TrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps_per_epoch * args.epochs)),
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="no",
        bf16=bf16,
        fp16=use_cuda and not bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=use_4bit,
        gradient_checkpointing_kwargs=gc_kwargs if use_4bit else None,
        report_to="none",
        seed=config.SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()

    out = args.output_dir
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with open(f"{out}/training_info.json", "w") as f:
        json.dump({"base_model": args.base_model, "train_examples": len(train_ds), "epochs": args.epochs,
                   "lr": args.lr, "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA,
                   "log_history": trainer.state.log_history}, f, indent=2)
    plot_loss(trainer.state.log_history)
    print(f"Saved LoRA adapter -> {out}")


def plot_loss(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*tr), label="train loss")
    if ev:
        ax.plot(*zip(*ev), "o-", label="validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("QLoRA fine-tuning loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "training_loss.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
