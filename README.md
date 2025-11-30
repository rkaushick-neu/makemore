# Makemore: Character Level Language Models

A step by step approach to building language models from scratch.

This repository is built to replicate the existing [makemore project](https://github.com/karpathy/makemore) built by Andrej Karpathy.

## Names from the dataset

Example names from the dataset:

```
['emma',
 'olivia',
 'ava',
 'isabella',
 'sophia',
 'charlotte',
 'mia',
 'amelia',
 'harper',
 'evelyn'
 ]
```

## Bigram character model

Bigram models predict the next character based on the previous character.

### Building the bigram matrix N

The bigram matrix `N` is a 27×27 matrix that counts how many times each character follows another character in the dataset. It consists of 26 letters (a-z) and 1 special token `.` used to mark both the start and end of words.

#### Creating character mappings

First, we create lookups from character to number and vice versa:

```python
chars = sorted(list(set(''.join(words))))
# s to i: mapping from character to integer
stoi = {s: i+1 for i, s in enumerate(chars)}
# special token for both start and end of word
stoi['.'] = 0
# reverse the dictionary stoi --> itos
itos = {i: s for s,i in stoi.items()}
```

In the above code, `stoi` (string to integer) would be as follows:

```
stoi = 
{
 '.': 0,
 'a': 1,
 'b': 2,
 'c': 3,
    ...
 'z': 26}
```

`itos` (integer to string) is simply the reverse of `stoi`.

#### Populating the matrix N

For each word, we create character pairs (bigrams) including the special `.` token at the start and end, then increment the corresponding position in matrix `N`:

```python
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        # now for both the characters get the index
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # now increment the 2D tensor N
        N[ix1, ix2] += 1
```

Each entry `N[i, j]` represents the count of how many times character `j` follows character `i` in the dataset. For example:
- `N[0, 1]` counts how many times 'a' appears at the start of a word (after '.')
- `N[1, 2]` counts how many times 'b' follows 'a'
- `N[5, 0]` counts how many times a word ends after 'e' ('.' follows 'e')

The final `N` matrix looks like the following:

![Bigram Matrix N](./images/bigram_matrix_N.png)

## Sampling words using multinomial distribution

To generate words from the bigram model, we sample characters sequentially using a multinomial distribution. The key optimization is to **pre-compute the normalized probability matrix** instead of normalizing probabilities inside the sampling loop.

### Efficient approach: Pre-normalize the probability matrix

Instead of normalizing the probability distribution in each iteration of the loop, we can normalize all rows of the count matrix `N` upfront:

```python
# Convert count matrix to probability matrix
P = N.float()
# Normalize each row (each row represents probability distribution for next character)
P /= P.sum(1, keepdim=True)
```

**Important:** Using `keepdim=True` is crucial for proper broadcasting. This ensures that:
- `P` has shape `[27, 27]`
- `P.sum(1, keepdim=True)` has shape `[27, 1]` (not `[27]`)
- Broadcasting works correctly to normalize each row independently

Without `keepdim=True`, the shape would be `[27]`, which would still broadcast but produce incorrect results (column-wise normalization instead of row-wise).

### Sampling loop

Once `P` is pre-normalized, the sampling loop becomes efficient:

```python
import torch

# Pre-normalize probability matrix
P = N.float()
P /= P.sum(1, keepdim=True)

# Generator for reproducibility
g = torch.Generator().manual_seed(2147483647)

# Generate multiple names
for i in range(20):
    out = []
    ix = 0  # Start with '.' (index 0)
    while True:
        # Use pre-computed probability distribution (already normalized)
        p = P[ix]
        # Sample next character index
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # '.' marks the end of the word
            break
    print(''.join(out))
```

### Why this is efficient

The naive approach would normalize probabilities inside the loop:

```python
while True:
    p = N[ix].float()
    p = p / p.sum()  # Normalizing inside loop - inefficient!
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    # ...
```

By pre-normalizing `P`, we:
1. **Reduce redundant computations**: Normalization happens once instead of once per character during sampling
2. **Improve performance**: Especially noticeable when generating many words
3. **Maintain correctness**: Each row of `P` is a proper probability distribution (sums to 1)