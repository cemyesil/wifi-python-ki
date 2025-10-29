#!/usr/bin/env python3
import sys, threading
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # small, ungated; swap to 3.1-8B if you have VRAM

SYSTEM = "You are a helpful, concise assistant."

def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # fp16 on GPU/MPS, fp32 on CPU; device_map="auto" places layers for you
    kwargs = {}
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        kwargs.update(dict(torch_dtype=torch.float16, device_map="auto"))
    else:
        kwargs.update(dict(torch_dtype=torch.float32))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model

def format_chat(tok, history: List[Dict[str, str]], user_text: str):
    msgs = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate_stream(tok, model, prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.95):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, do_sample=True, temperature=temperature,
                  top_p=top_p, max_new_tokens=max_new_tokens,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    return streamer

def main():
    tok, model = load_model(MODEL_ID)
    print("Chat ready. Commands: /reset, /exit")
    history: List[Dict[str, str]] = []

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not user: continue
        if user.lower() in ("/exit", "/quit"): print("Bye."); break
        if user.lower() == "/reset": history.clear(); print("[history cleared]"); continue

        prompt = format_chat(tok, history, user)
        print("Assistant: ", end="", flush=True)
        reply_parts = []
        for chunk in generate_stream(tok, model, prompt):
            reply_parts.append(chunk); sys.stdout.write(chunk); sys.stdout.flush()
        reply = "".join(reply_parts).strip() or "(no response)"
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]

if __name__ == "__main__":
    main()
