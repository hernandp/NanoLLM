# NanoLLM

NanoLLM is a deliberately small educational project for learning the pieces that
sit underneath a language model pipeline. The codebase is not a full LLM
framework. Instead, it focuses on a few core stages that are small enough to
read end-to-end:

- Byte Pair Encoding (BPE) vocabulary training
- text encoding with saved merge rules
- a tiny token-embedding trainer
- built-in self-tests

The project is written to be understandable first. The newer training code is
commented heavily on purpose so the implementation can double as a tutorial.

## What The Project Does

The current workflow looks like this:

1. Train a BPE tokenizer from raw text with `/bpe`
2. Reuse that tokenizer to encode other text with `/encode`
3. Train token embeddings with `/train`
4. Validate the critical functionality with `/selftest`

Each step produces plain text artifacts so you can inspect them manually.

## Building

This repository is set up as a Visual Studio C++ project.

### Visual Studio

Open `NanoLLM.sln` and build the `Debug|x64` or `Release|x64` configuration.

### MSBuild

Example:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe" `
    NanoLLM.sln /t:Build /p:Configuration=Debug /p:Platform=x64
```

The executable is typically generated at:

```text
x64\Debug\NanoLLM.exe
```

## Command Line Usage

### Help

```text
NanoLLM.exe /?
```

### License

```text
NanoLLM.exe /lic
```

### BPE Vocabulary Training

```text
NanoLLM.exe /bpe src=corpus.txt out=tokenizer m=1000
```

This writes:

- `tokenizer.voc`
- `tokenizer.merges`

### Encoding Text

```text
NanoLLM.exe /encode mergefile=tokenizer input=text.txt output=encoded
```

This writes:

- `encoded.tok`

If `input=` is omitted, text is read from standard input until EOF.

If `output=` is omitted, the encoded tokens are printed to the console.

### Embedding Training

```text
NanoLLM.exe /train mergefile=tokenizer input=corpus.txt output=model dim=64 window=4 epochs=3 neg=5 lr=0.025 seed=1
```

This writes:

- `model.emb`
- `model.meta`

### Self-Tests

```text
NanoLLM.exe /selftest
```

## Recommended Workflow

The intended learning flow is:

1. Build a tokenizer from a raw corpus:

```text
NanoLLM.exe /bpe src=corpus.txt out=tokenizer m=1000
```

2. Inspect the generated files:

- `tokenizer.voc`
- `tokenizer.merges`

3. Encode sample text:

```text
NanoLLM.exe /encode mergefile=tokenizer input=text.txt output=encoded
```

4. Train token embeddings from the same corpus:

```text
NanoLLM.exe /train mergefile=tokenizer input=corpus.txt output=model
```

5. Inspect the learned embedding file:

- `model.emb`
- `model.meta`

## Internals: The BPE Stage

The BPE implementation in this project is intentionally small and readable.

### High-level idea

It starts with characters and repeatedly merges the most frequent adjacent token
pair.

For example, a word like `lowest` begins as:

```text
l o w e s t </w>
```

If `o + w` is common, that pair may be merged into:

```text
l ow e s t </w>
```

As training continues, larger subword pieces can appear.

### How NanoLLM applies BPE

1. Normalize the corpus to lowercase alphabetic words
2. Split each word into one-character tokens
3. Append `</w>` so merges do not cross word boundaries
4. Count adjacent token pairs
5. Merge the most frequent pair
6. Repeat until the target vocabulary size is reached or no strong merge remains

### Output files

#### `.voc`

The vocabulary file contains one token per line.

It includes:

- `<PAD>`
- `<END>`
- learned character/subword tokens
- merged word-ending tokens such as `the</w>`

#### `.merges`

The merge file records each learned merge in training order, for example:

```text
t + h -> th
th + e</w> -> the</w>
```

Encoding replays these merges in order. That is why the merge order matters.

### Special tokens

- `<PAD>` is intended for future padding use
- `<END>` is intended for future end-of-sequence use
- `</w>` marks word boundaries during BPE training and encoding

### Current simplifications

The tokenizer is still intentionally simplified:

- it lowercases text
- it keeps alphabetic words only
- punctuation and numbers are not yet first-class tokens

That behavior also affects encoding and training, because later stages reuse the
same tokenization logic.

## Internals: The Training Stage

The `/train` command learns a small embedding vector for each token in the BPE
vocabulary.

### What an embedding is

An embedding is a short numeric vector assigned to each token. The goal is to
place tokens used in similar contexts near each other in vector space.

A later model could use these vectors as its first learned representation of
text.

### What the trainer does

This project uses a minimal **skip-gram with negative sampling** setup, inspired
by word2vec.

The core idea is:

- pick one token as the center token
- look at nearby tokens inside a context window
- train the center token embedding to predict those nearby tokens

### Context window

If the token stream is:

```text
the</w> low er</w> dog</w>
```

and the window size is `2`, then a token can learn from nearby tokens up to two
positions away on each side.

### Why negative sampling exists

A full softmax over the entire vocabulary would be unnecessarily large for a
tiny educational trainer. Negative sampling is a cheaper approximation.

For each real `(center, context)` pair:

- train once with label `1` for the real context token
- train several times with label `0` for random noise tokens

This teaches the model:

- which token pairs should score high
- which token pairs should score low

### Why there are two embedding tables during training

Classic skip-gram training keeps:

- an **input** embedding table for center tokens
- an **output/context** embedding table for predicted context tokens

Both are updated during training. After training, NanoLLM exports only the input
table as the final token embedding space.

### Training data flow

The `/train` pipeline is:

```text
raw corpus text
-> BPE encoding with existing .merges
-> token strings
-> token ids from .voc
-> skip-gram training pairs
-> SGD updates
-> exported embedding matrix
```

### Output files

#### `.emb`

This is the learned embedding table in text form.

The first line is a header:

```text
tokens=<N> dim=<D>
```

Each remaining line contains:

```text
<token>    <v1>    <v2>    ...
```

#### `.meta`

This stores metadata about the training run, including:

- tokenizer base name
- corpus path
- dimension
- context window
- epochs
- negative samples
- learning rate
- seed
- encoded token count
- elapsed time

## Testing

The project includes a built-in self-test mode:

```text
NanoLLM.exe /selftest
```

The self-tests currently cover:

- named argument parsing
- BPE vocabulary and merge generation
- encoding behavior
- vocabulary / merge / token file round-trips
- embedding trainer smoke tests
- deterministic training with a fixed seed
- embedding and metadata file generation

## Limitations

This is still a very small educational approximation.

Current limitations include:

- no transformer model yet
- no batching or multithreaded training
- no checkpoint resume support
- no binary model format
- tokenizer behavior is still simplified
- embedding training is intentionally plain SGD without advanced optimizers

## Reasonable Next Steps

Possible future expansions:

- tokenize punctuation and numbers explicitly
- add a binary model format for faster loading
- add a command to inspect or query learned embeddings
- add a tiny forward-pass language model on top of the embeddings
- introduce simple positional embeddings and transformer blocks later

For now, the project is best viewed as a transparent, inspectable path from raw
text to subword tokens to learned token vectors.
