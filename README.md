# nnx-lm: A portable, pip-installable CLI for running LLMs via JAX on any hardware backend.

```fish
pip install nnx-lm==0.0.1
nlm
```

```python
import nnxlm
new_str, new_ids = nnxlm.generate(model_id='Qwen/Qwen3-0.6B', model_cls=nnxlm.Qwen3ForCausalLM)
```

```
=== Input ===
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

=== Output===
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they're trained on massive datasets, so they can learn a lot. Then talk about their capabilities, like understanding context, generating coherent responses, and being able to handle various tasks. Also, mention that they're not just text

=== Benchmarks ===
Prompt processing: 219.4 tokens/sec (18 tokens in 0.1s)
Token generation: 24.9 tokens/sec (100 tokens in 4.0s)
```

Batch:
```python
new_str, new_ids = nnxlm.generate(model_id='Qwen/Qwen3-0.6B', model_cls=nnxlm.Qwen3ForCausalLM, use_chat_template=False)
```

```
=== Input ===
#write a quick sort algorithm

=== Output===
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

arr = [5, 3, 8, 1, 4, 2]
print(quick_sort(arr))

#this code is not working

=== Input ===
#hello world in C

=== Output===
I need to write a C program that outputs "Hello World" in the console. I have to write the code in C, and I need to make sure that the program is correct. I have to write the code in a file named hello.c. I need to make sure that the code is correct. I have to write the code in a file named hello.c. I need to make sure that the code is correct. I have to write the code in a file named hello.c.

=== Benchmarks ===
Prompt processing: 188.8 tokens/sec (14 tokens in 0.1s)
Token generation: 45.1 tokens/sec (200 tokens in 4.4s)
```

Scan:
```python
new_str, new_ids = nnxlm.generate(model_id='Qwen/Qwen3-0.6B', model_cls=nnxlm.Qwen3ForCausalLM, use_scan=True)
```

```
=== Input ===
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

=== Output===
<think>
Okay, the user wants an introduction to a large language language model. Let me start by remembering the correct answer. The correct answer is in the box. The box is the correct answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is the answer. The correct answer is

=== Benchmarks ===
Prompt processing: 28.7 tokens/sec (18 tokens in 0.6s)
Token generation: 74.4 tokens/sec (100 tokens in 1.3s)
```

