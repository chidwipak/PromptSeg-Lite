"""Prompt text pools for text-conditioned segmentation."""

PROMPT_POOL = {
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment drywall joint",
        "segment tape line",
    ],
    "crack": [
        "segment crack",
        "segment wall crack",
        "segment surface crack",
        "segment fracture",
        "segment crack line",
    ],
}

# All unique characters across all prompts (for character-level tokenization)
ALL_PROMPTS = []
for pool in PROMPT_POOL.values():
    ALL_PROMPTS.extend(pool)

VOCAB = sorted(set("".join(ALL_PROMPTS)))  # unique characters
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 reserved for padding
CHAR_TO_IDX["<PAD>"] = 0
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHAR_TO_IDX)
MAX_PROMPT_LEN = max(len(p) for p in ALL_PROMPTS)


def tokenize(text, max_len=None):
    """Convert text to list of character indices."""
    if max_len is None:
        max_len = MAX_PROMPT_LEN
    tokens = [CHAR_TO_IDX.get(c, 0) for c in text]
    # Pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens
