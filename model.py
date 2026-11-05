"""The real model behind the walkthrough.

This file does three things, and the page shows each one:
  1. tokenize()  turns the prompt into token ids (numbers the model can read)
  2. forward()   runs the model once over those ids and scores every possible next token
  3. info()      describes the model, for the header of the page

It keeps no memory between calls. The browser holds the list of token ids and sends the
whole list every time, so each pass reruns the network over the full input. That is
exactly what the demo teaches. Real chat servers save past work instead (the "KV cache").
"""
import torch  # PyTorch: the maths library the model runs on
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face: downloads and runs models

# A small chat model (about 500 million numbers inside it), small enough for a laptop CPU.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Load once, when this file is imported. The first run downloads about 1 GB.
# ---------------------------------------------------------------------------
print("Loading model (first run downloads about 1 GB)...")

# The tokenizer is the dictionary: text <-> token ids.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# The model itself. "eager" attention is slower, but it is the only kind that can
# hand back its attention weights, which the page draws on the tokens.
llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
llm.eval()  # switch off training-only behaviour; we only ever ask it questions

# The ids that mean "I am done answering". The page stops when the model picks one.
_stop = llm.generation_config.eos_token_id
_stop = _stop if isinstance(_stop, list) else [_stop]
STOP_IDS = sorted(set(_stop + [tokenizer.eos_token_id]))


def info():
    """A few facts about the model, shown at the top of the page."""
    return {
        "model": MODEL_ID,
        "vocab_size": len(tokenizer),  # how many different tokens it knows
        "layers": llm.config.num_hidden_layers,  # how many blocks the input passes through
        "heads": llm.config.num_attention_heads,  # attention "heads" per block
        "params_millions": round(sum(p.numel() for p in llm.parameters()) / 1e6),
    }


def token_text(token_id):
    """The piece of text one token id stands for, e.g. 785 -> 'The'."""
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def tokenize(prompt, use_template):
    """Turn the prompt into token ids, each marked hidden or visible.

    A chat model never sees your words alone. They are wrapped in a "chat template":
    hidden text that says who is speaking. Those tokens are marked hidden, because the
    student never typed them.
    """
    if not use_template:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return [{"id": i, "text": token_text(i), "hidden": False} for i in ids]

    # The full text the model reads: template + prompt + template.
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    # Split it into the part before the prompt, the prompt, and the part after.
    before, _, after = full.partition(prompt)

    tokens = []
    for part, hidden in [(before, True), (prompt, False), (after, True)]:
        for i in tokenizer(part, add_special_tokens=False)["input_ids"]:
            tokens.append({"id": i, "text": token_text(i), "hidden": hidden})
    return tokens


@torch.no_grad()  # we are not training, so skip the bookkeeping training needs
def forward(ids, temperature, top_k, seed):
    """One pass of the model over `ids`. Returns its top guesses for the next token and its pick."""

    # Step 1: run the network. The input is a batch of one sentence, hence [ids].
    out = llm(torch.tensor([ids]), output_attentions=True)

    # Step 2: take the scores ("logits") for the next token: one number per word in
    # the vocabulary. [0, -1] means first sentence, last position.
    logits = out.logits[0, -1]

    # Step 3: turn scores into probabilities that add up to 1 (softmax), then pick one.
    if temperature <= 0:
        # Temperature 0: always take the most likely token ("greedy").
        probs = torch.softmax(logits, dim=-1)
        pick = int(torch.argmax(logits))
    else:
        # Higher temperature flattens the odds, so less likely tokens get a chance.
        # The seed makes the dice roll repeatable, so a class sees the same result twice.
        probs = torch.softmax(logits / temperature, dim=-1)
        dice = torch.Generator().manual_seed(seed)
        pick = int(torch.multinomial(probs, 1, generator=dice))

    # Step 4: the top few candidates, for the probability bars on the page.
    top_probs, top_ids = torch.topk(probs, top_k)
    candidates = [
        {"id": int(i), "text": token_text(int(i)), "prob": float(p), "logit": float(logits[i])}
        for p, i in zip(top_probs, top_ids)
    ]

    # Step 5: attention. For each layer, how much the last token "looked at" every
    # earlier token, averaged over the heads. The page colours the tokens with this.
    attention = [layer[0, :, -1, :].mean(dim=0).tolist() for layer in out.attentions]

    return {
        "candidates": candidates,
        "pick": {"id": pick, "text": token_text(pick), "prob": float(probs[pick])},
        "attention": attention,
        "eos_ids": STOP_IDS,
    }
