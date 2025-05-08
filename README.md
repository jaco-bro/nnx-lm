# nnx-lm: A portable, pip-installable CLI for running LLMs via JAX on any hardware backend.

## Quick Start

```fish
pip install nnx-lm
nlm
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
nlm --scan
```

```
=== Input ===
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

=== Output===
<think>
Okay, the user wants a short introduction to a large language model. Let me start by defining what they mean. They might be a student or someone new to the field, or maybe they just want a simple intro.

I should avoid technical terms and instead focus on the key features. Let's see, the introduction should highlight the advantages of the current understanding.

The user's example is a bit of the structure where the assistant has to generate a sentence.

So, the answer is

=== Benchmarks ===
Prompt processing: 28.3 tokens/sec (18 tokens in 0.6s)
Token generation: 76.0 tokens/sec (100 tokens in 1.3s)
```

Batch:

```fish
nlm --no-format
```

```
=== Input ===
#write a quick sort algorithm

=== Output===
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)

arr = [5, 3, 8, 1, 4, 2]
print(quicksort(arr))

#this code is not working

=== Input ===
Give me a short introduction to large language model.

=== Output===
Large language models (LLMs) are artificial intelligence models that can understand and generate human language. They are trained on vast amounts of text data to understand and generate human language. LLMs are used in various applications, such as chatbots, translation, and content creation. They are also used in other areas like customer service, customer support, and even in creative writing. LLMs are becoming more advanced and are capable of understanding and generating more complex language. They are also being used in research

=== Benchmarks ===
Prompt processing: 31.6 tokens/sec (20 tokens in 0.6s)
Token generation: 45.0 tokens/sec (200 tokens in 4.4s)
```

Batched scan:

```fish
nlm --no-format --scan
```

```
=== Benchmarks ===
Prompt processing: 32.0 tokens/sec (20 tokens in 0.6s)
Token generation: 135.7 tokens/sec (200 tokens in 1.5s)
```

Jit:

```fish
nlm --jit
```

```
UserWarning: Some donated buffers were not usable: ShapedArray(int32[1,1]), ShapedArray(float32[1,1]), ShapedArray(bfloat16[1,1,1,118]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bfloat16[1,8,118,128]), ShapedArray(bool[1,1]).
Donation is not implemented for ('METAL',).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"

<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they're trained on massive datasets. Then talk about their capabilities, like understanding context and generating coherent responses. Also, highlight their applications in various fields. Oh, and maybe mention that they're not just text generators but can

=== Input ===
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

=== Output===
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they're trained on massive datasets. Then talk about their capabilities, like understanding context and generating coherent responses. Also, highlight their applications in various fields. Oh, and maybe mention that they're not just text generators but can handle

=== Benchmarks ===
Prompt processing: 28.3 tokens/sec (18 tokens in 0.6s)
Token generation: 18.0 tokens/sec (100 tokens in 5.6s)
```

Python:

```python
import nnxlm as nl
m = nl.load('Qwen/Qwen3-0.6B')
nl.generate(*m)
```

Test:

```python
nl.main.test()
```
