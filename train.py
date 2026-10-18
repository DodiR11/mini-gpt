import argparse
import random
from pathlib import Path

import torch

from config import (
    BATCH_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER,
    LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, SEED
)
from model import GPTLanguageModel


def main(data_path):
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    text = Path(data_path).read_text(encoding="utf-8")
    # build the character vocabulary from the dataset
    chars = sorted(set(text))
    vocab_size = len(chars)
    # map each character to an integer ID
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor(
        [stoi[ch] for ch in text],
        dtype=torch.long
    )

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        source = train_data if split == "train" else val_data
         # randomly choose starting positions for each sequence
        ix = torch.randint(
            len(source) - BLOCK_SIZE,
            (BATCH_SIZE,)
        )

        x = torch.stack([
            source[i:i + BLOCK_SIZE] for i in ix
        ])

        y = torch.stack([
            source[i + 1:i + BLOCK_SIZE + 1] for i in ix
        ])

        return x.to(device), y.to(device)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
    # switch to evaluation mode without calculating gradients
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}

        for split in ("train", "val"):
            values = torch.zeros(EVAL_ITERS)

            for k in range(EVAL_ITERS):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                values[k] = loss.item()
              # average the loss over multiple batches
            losses[split] = values.mean().item()

        model.train()
        return losses

    checkpoint_dir = (
        Path(__file__).resolve().parent / "checkpoints"
    )
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss()

            print(
                f"Step {step}: "
                f"train {losses['train']:.4f}, "
                f"val {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "chars": chars,
                    "hyperparameters": {
                        "block_size": BLOCK_SIZE,
                        "n_embd": N_EMBD,
                        "n_head": N_HEAD,
                        "n_layer": N_LAYER
                    }
                }, checkpoint_dir / "best_model.pt")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        # reset gradients, calculate new ones, and update weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete!")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mini-GPT on a text file"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to a UTF-8 training text file"
    )

    args = parser.parse_args()
    main(args.data)
