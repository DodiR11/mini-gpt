import argparse
from pathlib import Path

import torch

from config import BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER
from model import GPTLanguageModel


def generate_text(prompt="Tom", max_new_tokens=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "best_model.pt"
    )
   # load the saved model checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
  # rebuild the character mappings used during training
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = GPTLanguageModel(
        vocab_size=len(chars),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
# check if the prompt contains unknown characters
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(
            f"Prompt contains characters outside the vocabulary: {unknown}"
        )

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    context = torch.tensor(
        [[stoi[ch] for ch in prompt]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens
        )

    return "".join(
        itos[i] for i in generated_ids[0].tolist()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using Mini-GPT"
    )

    parser.add_argument(
        "--prompt",
        default="Tom"
    )

    parser.add_argument(
        "--length",
        type=int,
        default=300
    )

    args = parser.parse_args()

    print(generate_text(args.prompt, args.length))
