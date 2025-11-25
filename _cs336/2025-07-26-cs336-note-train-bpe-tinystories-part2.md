---
layout: post
title: "Study Notes: Stanford CS336 Language Modeling from Scratch [3b] - Building BPE Tokenizer (Part 2)"
categories: cs336
author:
- 大模型我都爱
---

<style>
  .xiaohongshu-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #ff2442; /* 小红书主色 */
    text-decoration: none;
    font-weight: bold;
    font-size: 14px;
  }
  .xiaohongshu-link:hover {
    text-decoration: underline;
  }
  .xiaohongshu-logo {
    width: 18px;
    height: 18px;
    border-radius: 4px;
  }
</style>

<div style="padding:12px;border:1px solid #eee;border-radius:8px;display:inline-block;margin-bottom:20px;">
  <strong>大模型我都爱</strong><br>
  <p style="margin:4px 0;">
    小红书号：
    <a class="xiaohongshu-link"
       href="https://www.xiaohongshu.com/user/profile/5b2c5758e8ac2b08bf20e38d"
       target="_blank">
      <img class="xiaohongshu-logo"
           src="https://static.cdnlogo.com/logos/r/77/rednote-xiaohongshu.svg"
           alt="小红书 logo">
      119826921
    </a>
  </p>
  IP属地：美国
</div>

# Building a BPE Tokenizer from Scratch: Training Results and Testing (Part 2)

# 从零开始构建BPE分词器：训练结果和测试（第2部分）

This is Part 2 of building a BPE tokenizer. See [Part 1](/chinese-online-content-for-llm/cs336/cs336-note-train-bpe-tinystories-part1/) for the implementation details including the chunking algorithm and BPE training code.

这是构建BPE分词器的第2部分。查看[第1部分](/chinese-online-content-for-llm/cs336/cs336-note-train-bpe-tinystories-part1/)了解实现细节，包括分块算法和BPE训练代码。

## Training on TinyStories Dataset

## 在TinyStories数据集上训练

Now let's use our implementation to train a tokenizer on the TinyStories dataset. Here is one training function to demonstrate all the steps:

现在让我们使用我们的实现在TinyStories数据集上训练分词器。这是一个演示所有步骤的训练函数：

```python
import time
import os

def train_bpe_tokentizer_via_dataset(input_path: str):
    print("=" * 80)
    print("BPE TOKENIZER TRAINING ON TINYSTORIES DATASET")
    print("=" * 80)

    # Configuration
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        print("Please ensure the TinyStories dataset is in the data/ directory.")
        return

    # Display configuration
    file_size = os.path.getsize(input_path)
    print(f"Configuration:")
    print(f"  Input file: {input_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Target vocabulary size: {vocab_size:,}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Verbose logging: Enabled")
    print()

    # Train the tokenizer with verbose output
    overall_start_time = time.time()
    vocab, merges = train_bpe_tokenizer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        verbose=True  # Enable detailed logging
    )
    overall_end_time = time.time()

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total training time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Final vocabulary size: {len(vocab):,}")
    print(f"Number of merges performed: {len(merges):,}")
    print(f"Actual vocab size vs target: {len(vocab)} / {vocab_size}")

    # Save the tokenizer
    vocab_path = "tinystories_vocab.pkl"
    merges_path = "tinystories_merges.pkl"

    print(f"\nSaving tokenizer to disk...")
    save_tokenizer(vocab, merges, vocab_path, merges_path)
    print(f"  ✓ Vocabulary saved to: {vocab_path}")
    print(f"  ✓ Merges saved to: {merges_path}")

    # Detailed vocabulary analysis
    print("\n" + "=" * 80)
    print("VOCABULARY ANALYSIS")
    print("=" * 80)

    # Count different types of tokens
    byte_tokens = sum(1 for token_id in vocab.keys() if token_id < 256)
    special_token_count = len(special_tokens)
    merged_tokens = len(vocab) - byte_tokens - special_token_count

    print(f"Token type breakdown:")
    print(f"  Byte tokens (0-255): {byte_tokens}")
    print(f"  Special tokens: {special_token_count}")
    print(f"  Merged tokens: {merged_tokens}")
    print(f"  Total: {len(vocab)}")

    # Show some vocabulary examples
    print(f"\nByte tokens (first 10):")
    for i in range(10):
        if i in vocab:
            char = vocab[i].decode('utf-8', errors='replace')
            if char.isprintable() and char != ' ':
                print(f"  Token {i:3d}: {vocab[i]} -> '{char}'")
            else:
                print(f"  Token {i:3d}: {vocab[i]} -> {repr(char)}")

    print(f"\nSpecial tokens:")
    for token_str in special_tokens:
        token_bytes = token_str.encode('utf-8')
        for token_id, vocab_bytes in vocab.items():
            if vocab_bytes == token_bytes:
                print(f"  Token {token_id:3d}: {vocab_bytes} -> '{token_str}'")
                break

    print(f"\nMost recently merged tokens (last 10):")
    merged_token_ids = [tid for tid in sorted(vocab.keys()) if tid >= 256 + len(special_tokens)]
    for token_id in merged_token_ids[-10:]:
        try:
            decoded = vocab[token_id].decode('utf-8', errors='replace')
            print(f"  Token {token_id:4d}: {vocab[token_id]} -> '{decoded}'")
        except:
            print(f"  Token {token_id:4d}: {vocab[token_id]} -> (non-UTF8)")

    print(f"\nFirst 10 merge operations:")
    for i, (left, right) in enumerate(merges[:10]):
        try:
            left_str = left.decode('utf-8', errors='replace')
            right_str = right.decode('utf-8', errors='replace')
            merged_str = (left + right).decode('utf-8', errors='replace')
            print(f"  Merge {i+1:2d}: '{left_str}' + '{right_str}' -> '{merged_str}'")
        except:
            print(f"  Merge {i+1:2d}: {left} + {right} -> (binary)")

    print(f"\nLast 10 merge operations:")
    for i, (left, right) in enumerate(merges[-10:], len(merges) - 9):
        try:
            left_str = left.decode('utf-8', errors='replace')
            right_str = right.decode('utf-8', errors='replace')
            merged_str = (left + right).decode('utf-8', errors='replace')
            print(f"  Merge {i:2d}: '{left_str}' + '{right_str}' -> '{merged_str}'")
        except:
            print(f"  Merge {i:2d}: {left} + {right} -> (binary)")

    # Show file sizes
    vocab_size_bytes = os.path.getsize(vocab_path)
    merges_size_bytes = os.path.getsize(merges_path)
    print(f"\nOutput file sizes:")
    print(f"  Vocabulary file: {vocab_size_bytes:,} bytes ({vocab_size_bytes / 1024:.1f} KB)")
    print(f"  Merges file: {merges_size_bytes:,} bytes ({merges_size_bytes / 1024:.1f} KB)")
    print(f"  Total: {vocab_size_bytes + merges_size_bytes:,} bytes ({(vocab_size_bytes + merges_size_bytes) / 1024:.1f} KB)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("You can now use the trained tokenizer for encoding/decoding text.")
    print(f"Load with: vocab, merges = load_tokenizer('{vocab_path}', '{merges_path}')")
```

To run the training, one can try for example:

要运行训练，可以尝试例如：

```python
train_bpe_tokentizer_via_dataset(input_path="data/TinyStoriesV2-GPT4-train.txt")
```

And it will output the following info from the training process:

它将输出训练过程中的以下信息：

```
================================================================================
BPE TOKENIZER TRAINING ON TINYSTORIES DATASET
================================================================================
Configuration:
  Input file: /content/TinyStoriesV2-GPT4-train.txt
  File size: 2,227,753,162 bytes (2124.6 MB)
  Target vocabulary size: 10,000
  Special tokens: ['<|endoftext|>']
  Verbose logging: Enabled

Step 1: Setting up parallel processing...
Using 12 processes for parallel tokenization
Finding chunk boundaries aligned with special token: <|endoftext|>
Initial guess of the chunk boundaries: [0, 185646096, 371292192, 556938288, 742584384, 928230480, 1113876576, 1299522672, 1485168768, 1670814864, 1856460960, 2042107056, 2227753152]
Created 12 chunks for processing

Step 2: Pre-tokenizing text corpus...
Pre-tokenization completed in 66.52 seconds
Found 59,904 unique word types
Total token count: 536,592,162
Most common words:
  1. '.' -> 41,764,519 times
  2. ',' -> 23,284,331 times
  3. ' the' -> 20,828,576 times
  4. ' and' -> 19,475,966 times
  5. ' a' -> 15,063,529 times

Step 3: Training BPE with 9,743 merges...
Initial vocabulary size: 257 (256 bytes + 1 special tokens)
============================================================
Merge    1/9743: ' ' + 't' -> ' t' (freq: 63,482,199, time: 0.273s)
Merge    2/9743: 'h' + 'e' -> 'he' (freq: 63,341,860, time: 0.318s)
Merge    3/9743: ' ' + 'a' -> ' a' (freq: 47,465,635, time: 0.340s)
Merge    4/9743: ' ' + 's' -> ' s' (freq: 32,362,158, time: 0.340s)
Merge    5/9743: ' ' + 'w' -> ' w' (freq: 31,485,643, time: 0.327s)
Merge    6/9743: 'n' + 'd' -> 'nd' (freq: 28,922,386, time: 0.332s)
Merge    7/9743: ' t' + 'he' -> ' the' (freq: 28,915,024, time: 0.320s)
Merge    8/9743: 'e' + 'd' -> 'ed' (freq: 24,836,456, time: 0.317s)
Merge    9/9743: ' ' + 'b' -> ' b' (freq: 22,147,488, time: 0.326s)
Merge   10/9743: ' t' + 'o' -> ' to' (freq: 20,892,273, time: 0.322s)
Merge  100/9743: ' ha' + 'pp' -> ' happ' (freq: 3,147,884, time: 0.251s)
Merge  200/9743: ' s' + 'e' -> ' se' (freq: 1,410,130, time: 0.343s)
Merge  300/9743: ' s' + 'omet' -> ' somet' (freq: 790,510, time: 0.245s)
Merge  400/9743: ' g' + 'ot' -> ' got' (freq: 524,776, time: 0.338s)
Merge  500/9743: ' e' + 'ach' -> ' each' (freq: 369,637, time: 0.321s)
Merge  600/9743: 'l' + 'f' -> 'lf' (freq: 279,566, time: 0.230s)
Merge  700/9743: ' wal' + 'k' -> ' walk' (freq: 221,114, time: 0.237s)
Merge  800/9743: ' do' + 'll' -> ' doll' (freq: 177,602, time: 0.324s)
Merge  900/9743: ' ' + 'G' -> ' G' (freq: 147,699, time: 0.214s)
Merge 1000/9743: 'ec' + 't' -> 'ect' (freq: 127,288, time: 0.233s)
Merge 1100/9743: ' l' + 'ight' -> ' light' (freq: 108,006, time: 0.208s)
Merge 1200/9743: ' d' + 'in' -> ' din' (freq: 92,211, time: 0.225s)
Merge 1300/9743: ' picture' + 's' -> ' pictures' (freq: 80,416, time: 0.318s)
Merge 1400/9743: 'itt' + 'en' -> 'itten' (freq: 68,466, time: 0.235s)
Merge 1500/9743: 'A' + 'my' -> 'Amy' (freq: 59,829, time: 0.306s)
Merge 1600/9743: ' tal' + 'king' -> ' talking' (freq: 53,781, time: 0.330s)
Merge 1700/9743: 'b' + 'all' -> 'ball' (freq: 48,005, time: 0.309s)
Merge 1800/9743: ' k' + 'iss' -> ' kiss' (freq: 43,477, time: 0.318s)
...
Merge 8000/9743: ' mom' + 'mies' -> ' mommies' (freq: 879, time: 0.205s)
Merge 8100/9743: ' cryst' + 'als' -> ' crystals' (freq: 840, time: 0.299s)
Merge 8200/9743: ' playd' + 'ate' -> ' playdate' (freq: 809, time: 0.283s)
Merge 8300/9743: ' support' + 'ing' -> ' supporting' (freq: 778, time: 0.200s)
Merge 8400/9743: ' activ' + 'ity' -> ' activity' (freq: 747, time: 0.300s)
Merge 8500/9743: 'L' + 'izzy' -> 'Lizzy' (freq: 716, time: 0.284s)
Merge 8600/9743: 'er' + 'ing' -> 'ering' (freq: 691, time: 0.311s)
Merge 8700/9743: ' tid' + 'ied' -> ' tidied' (freq: 660, time: 0.308s)
Merge 8800/9743: 'f' + 'lowers' -> 'flowers' (freq: 633, time: 0.295s)
Merge 8900/9743: ' Gra' + 'nd' -> ' Grand' (freq: 609, time: 0.299s)
Merge 9000/9743: ' frustr' + 'ation' -> ' frustration' (freq: 584, time: 0.301s)
Merge 9100/9743: 'amil' + 'iar' -> 'amiliar' (freq: 561, time: 0.205s)
Merge 9200/9743: ' P' + 'retty' -> ' Pretty' (freq: 542, time: 0.310s)
Merge 9300/9743: ' sal' + 'on' -> ' salon' (freq: 521, time: 0.292s)
Merge 9400/9743: ' p' + 'ounced' -> ' pounced' (freq: 502, time: 0.196s)
Merge 9500/9743: ' pops' + 'ic' -> ' popsic' (freq: 485, time: 0.185s)
Merge 9600/9743: ' pain' + 'ful' -> ' painful' (freq: 469, time: 0.298s)
Merge 9700/9743: 'solut' + 'ely' -> 'solutely' (freq: 454, time: 0.308s)
============================================================
BPE training completed in 2731.72 seconds
Final vocabulary size: 10000
Total merges performed: 9743
Compression ratio: 4.07x (from 2,192,422,648 to 538,511,097 tokens)

================================================================================
TRAINING SUMMARY
================================================================================
Total training time: 2898.45 seconds
Final vocabulary size: 10,000
Number of merges performed: 9,743
Actual vocab size vs target: 10000 / 10000

Saving tokenizer to disk...
  ✓ Vocabulary saved to: tinystories_vocab.pkl
  ✓ Merges saved to: tinystories_merges.pkl

================================================================================
VOCABULARY ANALYSIS
================================================================================
Token type breakdown:
  Byte tokens (0-255): 256
  Special tokens: 1
  Merged tokens: 9743
  Total: 10000

Byte tokens (first 10):
  Token   0: b'\x00' -> '\x00'
  Token   1: b'\x01' -> '\x01'
  Token   2: b'\x02' -> '\x02'
  Token   3: b'\x03' -> '\x03'
  Token   4: b'\x04' -> '\x04'
  Token   5: b'\x05' -> '\x05'
  Token   6: b'\x06' -> '\x06'
  Token   7: b'\x07' -> '\x07'
  Token   8: b'\x08' -> '\x08'
  Token   9: b'\t' -> '\t'

Special tokens:
  Token 256: b'<|endoftext|>' -> '<|endoftext|>'

Most recently merged tokens (last 10):
  Token 9990: b' improving' -> ' improving'
  Token 9991: b' nicest' -> ' nicest'
  Token 9992: b' whiskers' -> ' whiskers'
  Token 9993: b' booth' -> ' booth'
  Token 9994: b' Land' -> ' Land'
  Token 9995: b'Rocky' -> 'Rocky'
  Token 9996: b' meadows' -> ' meadows'
  Token 9997: b' Starry' -> ' Starry'
  Token 9998: b' imaginary' -> ' imaginary'
  Token 9999: b' bold' -> ' bold'

First 10 merge operations:
  Merge  1: ' ' + 't' -> ' t'
  Merge  2: 'h' + 'e' -> 'he'
  Merge  3: ' ' + 'a' -> ' a'
  Merge  4: ' ' + 's' -> ' s'
  Merge  5: ' ' + 'w' -> ' w'
  Merge  6: 'n' + 'd' -> 'nd'
  Merge  7: ' t' + 'he' -> ' the'
  Merge  8: 'e' + 'd' -> 'ed'
  Merge  9: ' ' + 'b' -> ' b'
  Merge 10: ' t' + 'o' -> ' to'

Last 10 merge operations:
  Merge 9734: ' impro' + 'ving' -> ' improving'
  Merge 9735: ' nice' + 'st' -> ' nicest'
  Merge 9736: ' wh' + 'iskers' -> ' whiskers'
  Merge 9737: ' bo' + 'oth' -> ' booth'
  Merge 9738: ' L' + 'and' -> ' Land'
  Merge 9739: 'Rock' + 'y' -> 'Rocky'
  Merge 9740: ' meadow' + 's' -> ' meadows'
  Merge 9741: ' St' + 'arry' -> ' Starry'
  Merge 9742: ' imag' + 'inary' -> ' imaginary'
  Merge 9743: ' bo' + 'ld' -> ' bold'

Output file sizes:
  Vocabulary file: 117,701 bytes (114.9 KB)
  Merges file: 109,714 bytes (107.1 KB)
  Total: 227,415 bytes (222.1 KB)

================================================================================
TRAINING COMPLETED SUCCESSFULLY!
================================================================================
You can now use the trained tokenizer for encoding/decoding text.
Load with: vocab, merges = load_tokenizer('tinystories_vocab.pkl', 'tinystories_merges.pkl')
```

## Using the Trained Tokenizer

## 使用训练好的分词器

Once we have a trained tokenizer, we need a class to encode and decode text. Here's one complete implementation:

一旦我们有了训练好的分词器，我们需要一个类来编码和解码文本。这是一个完整的实现：

```python
class SimpleBPETokenizer:
    """Simple BPE tokenizer for encoding/decoding text."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  # {token_id: bytes}
        self.merges = merges  # [(left_bytes, right_bytes), ...]
        self.special_tokens = special_tokens or ["<|endoftext|>"]

        # Create reverse mapping for decoding
        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        # GPT-2 style regex pattern
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

        # Build merge rules for faster encoding
        self.merge_rules = {}
        for i, (left_bytes, right_bytes) in enumerate(merges):
            # Find what tokens these bytes correspond to
            left_id = self.bytes_to_id.get(left_bytes)
            right_id = self.bytes_to_id.get(right_bytes)
            merged_bytes = left_bytes + right_bytes
            merged_id = self.bytes_to_id.get(merged_bytes)

            if left_id is not None and right_id is not None and merged_id is not None:
                self.merge_rules[(left_id, right_id)] = merged_id

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if not text:
            return []

        # Handle special tokens
        token_ids = []
        remaining_text = text

        # Split on special tokens first
        for special_token in self.special_tokens:
            if special_token in remaining_text:
                parts = remaining_text.split(special_token)
                new_parts = []
                for i, part in enumerate(parts):
                    if i > 0:
                        # Add special token
                        special_bytes = special_token.encode('utf-8')
                        special_id = self.bytes_to_id.get(special_bytes)
                        if special_id is not None:
                            token_ids.append(special_id)
                    if part:
                        new_parts.append(part)
                remaining_text = ''.join(new_parts)

        # Apply regex tokenization
        for match in re.finditer(self.pattern, remaining_text):
            word = match.group()
            word_tokens = self._encode_word(word)
            token_ids.extend(word_tokens)

        return token_ids

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single word using BPE merges."""
        # Start with individual bytes
        word_bytes = word.encode('utf-8')
        tokens = []

        # Convert each byte to its token ID
        for byte_val in word_bytes:
            tokens.append(byte_val)  # Byte token IDs are 0-255

        # Apply BPE merges
        while len(tokens) > 1:
            # Find the best merge to apply
            best_merge = None
            best_pos = -1
            best_merge_priority = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rules:
                    # Find merge priority (earlier merges have higher priority)
                    merged_bytes = self.id_to_bytes[tokens[i]] + self.id_to_bytes[tokens[i + 1]]
                    for j, (left_bytes, right_bytes) in enumerate(self.merges):
                        if left_bytes + right_bytes == merged_bytes:
                            if j < best_merge_priority:
                                best_merge = pair
                                best_pos = i
                                best_merge_priority = j
                            break

            if best_merge is None:
                break

            # Apply the best merge
            new_tokens = tokens[:best_pos]
            new_tokens.append(self.merge_rules[best_merge])
            new_tokens.extend(tokens[best_pos + 2:])
            tokens = new_tokens

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Convert token IDs to bytes
        byte_sequences = []
        for token_id in token_ids:
            if token_id in self.id_to_bytes:
                byte_sequences.append(self.id_to_bytes[token_id])
            else:
                # Handle unknown tokens
                byte_sequences.append(b'<UNK>')

        # Concatenate all bytes and decode
        all_bytes = b''.join(byte_sequences)
        try:
            return all_bytes.decode('utf-8', errors='replace')
        except:
            return all_bytes.decode('utf-8', errors='ignore')

    def tokenize_with_details(self, text: str):
        """Tokenize text and show detailed breakdown."""
        token_ids = self.encode(text)

        print(f"Original text: '{text}'")
        print(f"Length: {len(text)} characters")
        print(f"UTF-8 bytes: {len(text.encode('utf-8'))} bytes")
        print(f"Token count: {len(token_ids)} tokens")
        print(f"Compression ratio: {len(text.encode('utf-8')) / len(token_ids):.2f}x")
        print()

        print("Token breakdown:")
        for i, token_id in enumerate(token_ids):
            token_bytes = self.id_to_bytes[token_id]
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
                if token_str.isprintable():
                    print(f"  {i+1:2d}. Token {token_id:4d}: '{token_str}' ({len(token_bytes)} bytes)")
                else:
                    print(f"  {i+1:2d}. Token {token_id:4d}: {repr(token_str)} ({len(token_bytes)} bytes)")
            except:
                print(f"  {i+1:2d}. Token {token_id:4d}: {token_bytes} (binary)")

        # Verify round-trip
        decoded = self.decode(token_ids)
        print(f"\nDecoded text: '{decoded}'")
        print(f"Round-trip successful: {text == decoded}")

        return token_ids
```

Let us compose some simple test cases below:

让我们编写一些简单的测试用例：

```python
def test_bpe_tokenizer():
    print("=" * 60)
    print("BPE TOKENIZER SAMPLE TESTS")
    print("=" * 60)

    # Load the trained tokenizer
    try:
        vocab, merges = load_tokenizer('tinystories_vocab.pkl', 'tinystories_merges.pkl')
        print(f"✓ Loaded tokenizer with {len(vocab)} vocab entries and {len(merges)} merges")
    except FileNotFoundError:
        print("Error: Tokenizer files not found!")
        print("Please run the training script first to create 'tinystories_vocab.pkl' and 'tinystories_merges.pkl'")
        return

    # Create tokenizer instance
    tokenizer = SimpleBPETokenizer(vocab, merges)
    print()

    # Example 1: Simple sentence
    print("EXAMPLE 1: Simple sentence")
    print("-" * 30)
    text1 = "Hello world! How are you today?"
    tokenizer.tokenize_with_details(text1)
    print()

    # Example 2: Text with special token
    print("EXAMPLE 2: Text with special token")
    print("-" * 30)
    text2 = "Once upon a time<|endoftext|>The end."
    tokenizer.tokenize_with_details(text2)
    print()

    # Example 3: Repeated words (should compress well)
    print("EXAMPLE 3: Repeated words")
    print("-" * 30)
    text3 = "the the the cat cat sat sat on on the the mat mat"
    tokenizer.tokenize_with_details(text3)
    print()

    # Example 4: Numbers and punctuation
    print("EXAMPLE 4: Numbers and punctuation")
    print("-" * 30)
    text4 = "I have 123 apples, 456 oranges, and 789 bananas!"
    tokenizer.tokenize_with_details(text4)
    print()

    # Example 5: Just encoding/decoding
    print("EXAMPLE 5: Simple encode/decode")
    print("-" * 30)
    text5 = "This is a test."
    token_ids = tokenizer.encode(text5)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Original: '{text5}'")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: '{decoded_text}'")
    print(f"Match: {text5 == decoded_text}")
    print()

    # Show some vocabulary statistics
    print("VOCABULARY STATISTICS")
    print("-" * 30)
    byte_tokens = sum(1 for tid in vocab.keys() if tid < 256)
    special_tokens = sum(1 for tid, token_bytes in vocab.items() if b'<|' in token_bytes)
    merged_tokens = len(vocab) - byte_tokens - special_tokens

    print(f"Byte tokens (0-255): {byte_tokens}")
    print(f"Special tokens: {special_tokens}")
    print(f"Merged tokens: {merged_tokens}")
    print(f"Total vocabulary: {len(vocab)}")

    # Show some example merged tokens
    print(f"\nSample merged tokens:")
    merged_token_ids = [tid for tid in sorted(vocab.keys()) if tid >= 257]
    for i, token_id in enumerate(merged_token_ids[:10]):
        token_bytes = vocab[token_id]
        try:
            decoded = token_bytes.decode('utf-8', errors='replace')
            print(f"  Token {token_id}: '{decoded}' ({len(token_bytes)} bytes)")
        except:
            print(f"  Token {token_id}: {token_bytes} (binary)")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
```

# BPE Tokenizer Sample Tests

# BPE分词器示例测试

Now run our complete test suite:

现在运行我们完整的测试套件：

```python
test_bpe_tokenizer()
```

Based on the training output from the TinyStories dataset, here are the testing results:

基于TinyStories数据集的训练输出，以下是测试结果：

✓ Loaded tokenizer with 10000 vocab entries and 9743 merges

✓ 加载了包含10000个词汇条目和9743次合并的分词器

## Example 1: Simple sentence

## 示例1：简单句子

**Original text:** 'Hello world! How are you today?'
**Length:** 31 characters
**UTF-8 bytes:** 31 bytes
**Token count:** 8 tokens
**Compression ratio:** 3.88x

**原始文本：** 'Hello world! How are you today?'
**长度：** 31个字符
**UTF-8字节：** 31字节
**词元数量：** 8个词元
**压缩比：** 3.88倍

### Token breakdown:

### 词元分解：

- Token 1183: 'Hello' (5 bytes)
- Token 1569: ' world' (6 bytes)
- Token 33: '!' (1 bytes)
- Token 2683: ' How' (4 bytes)
- Token 483: ' are' (4 bytes)
- Token 349: ' you' (4 bytes)
- Token 1709: ' today' (6 bytes)
- Token 63: '?' (1 bytes)

- 词元 1183: 'Hello' (5字节)
- 词元 1569: ' world' (6字节)
- 词元 33: '!' (1字节)
- 词元 2683: ' How' (4字节)
- 词元 483: ' are' (4字节)
- 词元 349: ' you' (4字节)
- 词元 1709: ' today' (6字节)
- 词元 63: '?' (1字节)

**Decoded text:** 'Hello world! How are you today?'
**Round-trip successful:** True

**解码文本：** 'Hello world! How are you today?'
**往返成功：** True

## Example 2: Text with special token

## 示例2：包含特殊标记的文本

**Original text:** 'Once upon a time<|endoftext|>The end.'
**Length:** 37 characters
**UTF-8 bytes:** 37 bytes
**Token count:** 8 tokens
**Compression ratio:** 4.62x

**原始文本：** 'Once upon a time<|endoftext|>The end.'
**长度：** 37个字符
**UTF-8字节：** 37字节
**词元数量：** 8个词元
**压缩比：** 4.62倍

### Token breakdown:

### 词元分解：

- Token 256: '`<|endoftext|>`' (13 bytes)
- Token 430: 'Once' (4 bytes)
- Token 439: ' upon' (5 bytes)
- Token 259: ' a' (2 bytes)
- Token 398: ' time' (5 bytes)
- Token 410: 'The' (3 bytes)
- Token 870: ' end' (4 bytes)
- Token 46: '.' (1 bytes)

- 词元 256: '`<|endoftext|>`' (13字节)
- 词元 430: 'Once' (4字节)
- 词元 439: ' upon' (5字节)
- 词元 259: ' a' (2字节)
- 词元 398: ' time' (5字节)
- 词元 410: 'The' (3字节)
- 词元 870: ' end' (4字节)
- 词元 46: '.' (1字节)

**Decoded text:** '<|endoftext|>Once upon a timeThe end.'
**Round-trip successful:** False

**解码文本：** '<|endoftext|>Once upon a timeThe end.'
**往返成功：** False

## Example 3: Repeated words

## 示例3：重复单词

**Original text:** 'the the the cat cat sat sat on on the the mat mat'
**Length:** 49 characters
**UTF-8 bytes:** 49 bytes
**Token count:** 13 tokens
**Compression ratio:** 3.77x

**原始文本：** 'the the the cat cat sat sat on on the the mat mat'
**长度：** 49个字符
**UTF-8字节：** 49字节
**词元数量：** 13个词元
**压缩比：** 3.77倍

### Token breakdown:

### 词元分解：

- Token 7199: 'the' (3 bytes)
- Token 263: ' the' (4 bytes)
- Token 263: ' the' (4 bytes)
- Token 459: ' cat' (4 bytes)
- Token 459: ' cat' (4 bytes)
- Token 1091: ' sat' (4 bytes)
- Token 1091: ' sat' (4 bytes)
- Token 354: ' on' (3 bytes)
- Token 354: ' on' (3 bytes)
- Token 263: ' the' (4 bytes)
- Token 263: ' the' (4 bytes)
- Token 1492: ' mat' (4 bytes)
- Token 1492: ' mat' (4 bytes)

- 词元 7199: 'the' (3字节)
- 词元 263: ' the' (4字节)
- 词元 263: ' the' (4字节)
- 词元 459: ' cat' (4字节)
- 词元 459: ' cat' (4字节)
- 词元 1091: ' sat' (4字节)
- 词元 1091: ' sat' (4字节)
- 词元 354: ' on' (3字节)
- 词元 354: ' on' (3字节)
- 词元 263: ' the' (4字节)
- 词元 263: ' the' (4字节)
- 词元 1492: ' mat' (4字节)
- 词元 1492: ' mat' (4字节)

**Decoded text:** 'the the the cat cat sat sat on on the the mat mat'
**Round-trip successful:** True

**解码文本：** 'the the the cat cat sat sat on on the the mat mat'
**往返成功：** True

## Example 4: Numbers and punctuation

## 示例4：数字和标点符号

**Original text:** 'I have 123 apples, 456 oranges, and 789 bananas!'
**Length:** 48 characters
**UTF-8 bytes:** 48 bytes
**Token count:** 19 tokens
**Compression ratio:** 2.53x

**原始文本：** 'I have 123 apples, 456 oranges, and 789 bananas!'
**长度：** 48个字符
**UTF-8字节：** 48字节
**词元数量：** 19个词元
**压缩比：** 2.53倍

### Token breakdown:

### 词元分解：

- Token 73: 'I' (1 bytes)
- Token 499: ' have' (5 bytes)
- Token 6314: ' 1' (2 bytes)
- Token 50: '2' (1 bytes)
- Token 51: '3' (1 bytes)
- Token 1836: ' apples' (7 bytes)
- Token 44: ',' (1 bytes)
- Token 9079: ' 4' (2 bytes)
- Token 53: '5' (1 bytes)
- Token 54: '6' (1 bytes)
- Token 5193: ' oranges' (8 bytes)
- Token 44: ',' (1 bytes)
- Token 267: ' and' (4 bytes)
- Token 32: ' ' (1 bytes)
- Token 55: '7' (1 bytes)
- Token 56: '8' (1 bytes)
- Token 57: '9' (1 bytes)
- Token 3898: ' bananas' (8 bytes)
- Token 33: '!' (1 bytes)

- 词元 73: 'I' (1字节)
- 词元 499: ' have' (5字节)
- 词元 6314: ' 1' (2字节)
- 词元 50: '2' (1字节)
- 词元 51: '3' (1字节)
- 词元 1836: ' apples' (7字节)
- 词元 44: ',' (1字节)
- 词元 9079: ' 4' (2字节)
- 词元 53: '5' (1字节)
- 词元 54: '6' (1字节)
- 词元 5193: ' oranges' (8字节)
- 词元 44: ',' (1字节)
- 词元 267: ' and' (4字节)
- 词元 32: ' ' (1字节)
- 词元 55: '7' (1字节)
- 词元 56: '8' (1字节)
- 词元 57: '9' (1字节)
- 词元 3898: ' bananas' (8字节)
- 词元 33: '!' (1字节)

**Decoded text:** 'I have 123 apples, 456 oranges, and 789 bananas!'
**Round-trip successful:** True

**解码文本：** 'I have 123 apples, 456 oranges, and 789 bananas!'
**往返成功：** True

## Example 5: Simple encode/decode

## 示例5：简单编码/解码

**Original:** 'This is a test.'
**Token IDs:** [1531, 431, 259, 2569, 46]
**Decoded:** 'This is a test.'
**Match:** True

**原始文本：** 'This is a test.'
**词元ID：** [1531, 431, 259, 2569, 46]
**解码文本：** 'This is a test.'
**匹配：** True

## Vocabulary Statistics

## 词汇统计

**Byte tokens (0-255):** 256
**Special tokens:** 1
**Merged tokens:** 9743
**Total vocabulary:** 10000

**字节词元（0-255）：** 256
**特殊词元：** 1
**合并词元：** 9743
**总词汇量：** 10000

### Sample merged tokens:

### 合并词元示例：

- Token 257: ' t' (2 bytes)
- Token 258: 'he' (2 bytes)
- Token 259: ' a' (2 bytes)
- Token 260: ' s' (2 bytes)
- Token 261: ' w' (2 bytes)
- Token 262: 'nd' (2 bytes)
- Token 263: ' the' (4 bytes)
- Token 264: 'ed' (2 bytes)
- Token 265: ' b' (2 bytes)
- Token 266: ' to' (3 bytes)

- 词元 257: ' t' (2字节)
- 词元 258: 'he' (2字节)
- 词元 259: ' a' (2字节)
- 词元 260: ' s' (2字节)
- 词元 261: ' w' (2字节)
- 词元 262: 'nd' (2字节)
- 词元 263: ' the' (4字节)
- 词元 264: 'ed' (2字节)
- 词元 265: ' b' (2字节)
- 词元 266: ' to' (3字节)

---
All examples completed successfully!

---
所有示例成功完成！
