# nnxlm
Run pretrained LLMs on any hardware (CPU, GPU, TPU, MPS, etc) in JAX using `flax.NNX` — no Torch, no HuggingFace Transformers (not even for the **tokenizers**).

## Supported Models

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen2.5-Coder-0.5B`
- `microsoft/Phi-4-mini-instruct`
- `ibm-granite/granite-3.3-2b-instruct`
- `THUDM/GLM-4-9B-0414`
- `HuggingFaceTB/SmolLM2-135M`
- `meta-llama/Llama-3.2-1B-Instruct`

All models run without PyTorch or `transformers`, using a custom tokenizer and model loader.

## Quick Start

```fish
pip install nnx-lm
nlm -p "Give me a short introduction to large language model.\n"
```

```
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they're trained on massive datasets, so they can learn a lot. Then talk about their capabilities, like understanding context, generating coherent responses, and being able to handle various tasks. Also, mention that they're not just

=== Input ===
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

=== Output===
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they're trained on massive datasets, so they can learn a lot. Then talk about their capabilities, like understanding context, generating coherent responses, and being able to handle various tasks. Also, mention that they're not just text

=== Benchmarks ===
Prompt processing: 28.4 tokens/sec (18 tokens in 0.6s)
Token generation: 22.8 tokens/sec (100 tokens in 4.4s)
```

## Examples

Scan:

```fish
nlm --scan -p "Give me a short introduction to large language model.\n"
```
- Prompt processing: 28.3 tokens/sec (18 tokens in 0.6s)
- Token generation: 76.0 tokens/sec (100 tokens in 1.3s)

Batch:

```fish
nlm -p "Give me a short introduction to large language model.\n"  "#write a quick sort algorithm\n"
```
- Prompt processing: 31.6 tokens/sec (20 tokens in 0.6s)
- Token generation: 45.0 tokens/sec (200 tokens in 4.4s)

Batched scan:

```fish
nlm --scan -p "Give me a short introduction to large language model.\n" "#write a quick sort algorithm\n"
```

- Prompt processing: 32.0 tokens/sec (20 tokens in 0.6s)
- Token generation: 135.7 tokens/sec (200 tokens in 1.5s)

Jit:

```fish
nlm --jit -p "Give me a short introduction to large language model.\n"
```

- Prompt processing: 28.3 tokens/sec (18 tokens in 0.6s)
- Token generation: 18.0 tokens/sec (100 tokens in 5.6s)

Python:

```python
import nnxlm as nl
m = nl.load('Qwen/Qwen3-0.6B')
nl.generate(*m, ["#write a quick sort algorithm\n", "Give me a short introduction to large language model.\n"])
```

Test:

```python
nl.main.test()
```
