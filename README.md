A small character level gpt model implemented in PyTorch and trained on 'The Adventures of Tom Sawyer by Mark Twain.

I built this project to practically understand how gpt-style language models work especially key concepts like self attention,Transformer blocks and autoregressive text generation.

I implemented the main components myself using basic pytorch layers Instead of using a pretrained model.

#Model Architecture

my model follows a decoder only transformer architecture.

It includes:
- tokenization on a character level 
- token embeddings
- positional embeddings
- Scaled dot product attention and causal masking
- multi-head self attention
- FF networks
- residual connections and layer normalization



#Configuration


| Parameter | Value |
|---|---|
| Vocabulary size | 75 |
| Context length | 64 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | AdamW |
| Training steps | 3,000 |
#Dataset

I trained the model on 'The Adventures of Tom Sawyer' that i obtained from Project Gutenberg(https://www.gutenberg.org/).

i cleaned then the text to remove the Project Gutenberg header and footer, and then split it into 90% training data and 10% validation data.

Since the model is character level, each unique character is mapped to an integer ID.Andthe model learns to predict the next character from the preceding sequence.

#Training

The model was trained using pytorch on my nvidia rtx 3050.

I used cross entropy loss for the next character prediction and then evaluated the model on the training and validation sets throughout training.

#Results

| Metric | Loss |
|---|---|
| Initial training loss | 4.1374 |
| Final training loss | 1.4436 |
| Initial validation loss | 4.1321 |
| Best validation loss | 1.5430 |


as we cam see the validation loss decreased steadily over the 3,000 training steps.I saved the best checkpoint and included it in the repository.

#Text Generation

The trained model generates text one character at a time by basically sampling from the predicted probability distribution.


an example output:


Tom, and the could tell ...
And her souls him, I won...
He was did not headle...


The generated text is obviously not fully coherent, but it has learned some patterns of English spelling, punctuation and sentence structure.

Given the small model size and the small dataset, I expected the model to learn local text patterns more successfully than long range meaning.



#A few things I would explore next:

- Subword tokenization instead of character-level tokenization
- A larger training dataset
- Longer context windows
- Dropout and learning rate scheduling
- top-k and temperature sampling for text generation

finally this project helped me connect the mathematical ideas behind attention with an actual working language model, from training to inference.

-References

- Attention Is All You Need (https://arxiv.org/abs/1706.03762)
- Andrej Karpathy, Let's build GPT(https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Mark Twain, 'The Adventures of Tom Sawyer',Project Gutenberg
