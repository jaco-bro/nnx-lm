from datetime import datetime
import os
import json
import time
import jax
import jax.numpy as jnp
from urllib.request import urlretrieve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Optional, Dict, Any, Union, Type, Callable
from flax import nnx
from safetensors.flax import load_file
from glob import glob
from tokenizerz import Tokenizer

def strftime_now(format="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(format)

def tqdm_hook(t):
    last_b = [0]
    def update_to(block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            t.total = total_size
        downloaded = block_num * block_size
        t.update(downloaded - last_b[0])
        last_b[0] = downloaded
    return update_to

def download_file(url, path, desc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"File '{path}' already exists. Skipping.")
        return
    with tqdm(unit='B', unit_scale=True, desc=desc, leave=False) as t:
        urlretrieve(url, path, reporthook=tqdm_hook(t))

def get_model_files(repo, model, dest=None):
    base_url = f"https://huggingface.co/{repo}/{model}/resolve/main"
    model_dir = model if dest is None else os.path.join(dest, model)
    os.makedirs(model_dir, exist_ok=True)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    try:
        if not os.path.exists(index_path):
            download_file(f"{base_url}/model.safetensors.index.json", index_path, "model index")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        pattern = next(iter(weight_map.values()))
        if "-of-" in pattern:
            base = pattern[:pattern.find("-00")]
            count = int(pattern.split("-of-")[1].split("-")[0].split(".")[0])
            ext = pattern[pattern.rfind("."):]
            files += [f"{base}-{i:05d}-of-{count:05d}{ext}" for i in range(1, count + 1)]
        else:
            files.append(pattern)
    except Exception:
        files.append("model.safetensors")
    return model_dir, [(f"{base_url}/{file}", os.path.join(model_dir, file), file) for file in files]

def download_model(repo, model, dest=None, max_workers=4):
    model_dir, tasks = get_model_files(repo, model, dest)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, path, desc) for url, path, desc in tasks]
        for future in futures:
            future.result()
    return model_dir

@dataclass
class Config:
    architectures: List[str]
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    eos_token_id: str
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    num_key_value_heads: int = None 
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "float32"
    head_dim: int = None
    attention_bias: bool = True
    mlp_bias: bool = False
    rope_traditional: bool = True
    partial_rotary_factor: float = 0.5
    max_position_embeddings: Optional[int] = None
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def dtype(self):
        return eval(f'jnp.{self.torch_dtype}')

def load_config(model_name, cls=Config):
    with open(f'{model_name}/config.json', 'r') as f:
        config_dict = json.load(f)
    return cls(**{k: v for k, v in config_dict.items() if k in {f.name for f in fields(cls)}})


class Roper(nnx.Module):
    def __init__(self, head_dim, theta=10000.0, traditional=False):
        dim = head_dim // 2
        self.freq = nnx.Variable(1.0 / (theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim)))

    @nnx.jit
    def __call__(self, positions):
        positions = positions[:, None, :, None]
        angles = positions * self.freq
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        return cos, sin

@jax.jit
def apply_rope(q, k, cos, sin):
    q_split = q.reshape(*q.shape[:-1], 2, -1)
    k_split = k.reshape(*k.shape[:-1], 2, -1)
    q_out = jnp.concatenate([
        q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
        q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
    ], axis=-1)
    k_out = jnp.concatenate([
        k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
        k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
    ], axis=-1)
    return q_out.astype(q.dtype), k_out.astype(k.dtype)

@jax.jit
def apply_partial_rope(q, k, cos, sin, partial_rotary_factor=0.5): # mock
    D = q.shape[-1]
    rot_D = D // 2
    q_rot, q_pass = q[..., :rot_D], q[..., rot_D:]
    k_rot, k_pass = k[..., :rot_D], k[..., rot_D:]
    q_pair = q_rot.reshape(*q_rot.shape[:-1], rot_D // 2, 2)
    k_pair = k_rot.reshape(*k_rot.shape[:-1], rot_D // 2, 2)
    q_even, q_odd = q_pair[..., 0], q_pair[..., 1]
    k_even, k_odd = k_pair[..., 0], k_pair[..., 1]
    cos_pair = cos[..., : rot_D // 2]
    sin_pair = sin[..., : rot_D // 2]
    q_rotated = jnp.stack(
        [q_even * cos_pair - q_odd * sin_pair,
         q_even * sin_pair + q_odd * cos_pair],
        axis=-1,
    ).reshape(q_rot.shape)
    k_rotated = jnp.stack(
        [k_even * cos_pair - k_odd * sin_pair,
         k_even * sin_pair + k_odd * cos_pair],
        axis=-1,
    ).reshape(k_rot.shape)
    q_out = jnp.concatenate([q_rotated, q_pass], axis=-1)
    k_out = jnp.concatenate([k_rotated, k_pass], axis=-1)
    return q_out.astype(q.dtype), k_out.astype(k.dtype)

@jax.jit
def create_causal_mask(padding_mask):
    padding_mask = jnp.array(padding_mask)
    seq_length = padding_mask.shape[1]
    causal_matrix = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
    causal_mask = jnp.where(causal_matrix & padding_mask[:, None, :], 0.0, -1e4)
    return causal_mask[:, None, :, :]

def measure_performance(start_time, prompt_time, end_time, batch_size, seq_length, gen_length):
    prompt_duration = prompt_time - start_time
    generation_duration = end_time - prompt_time
    tokens_processed = batch_size * seq_length
    tokens_generated = gen_length * batch_size
    prompt_throughput = tokens_processed / prompt_duration if prompt_duration > 0 else 0
    generation_throughput = tokens_generated / generation_duration if generation_duration > 0 else 0
    metrics = {
        "prompt_throughput": prompt_throughput,
        "generation_throughput": generation_throughput,
        "prompt_tokens": tokens_processed,
        "prompt_time": prompt_duration,
        "generation_tokens": tokens_generated,
        "generation_time": generation_duration
    }
    print('\n\n=== Benchmarks ===')
    print(f"Prompt processing: {prompt_throughput:.1f} tokens/sec ({tokens_processed} tokens in {prompt_duration:.1f}s)")
    print(f"Token generation: {generation_throughput:.1f} tokens/sec ({tokens_generated} tokens in {generation_duration:.1f}s)")
    return metrics

class Cache(nnx.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        self.max_len = max_len
        self.k = nnx.Variable(jnp.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype)) if k is None else nnx.Variable(k)
        self.v = nnx.Variable(jnp.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype)) if v is None else nnx.Variable(v)

    @nnx.jit
    def __call__(self, k, v):
        self.k.value = jnp.concat([self.k.value, k], axis=2)[:,:,-self.max_len:,:]
        self.v.value = jnp.concat([self.v.value, v], axis=2)[:,:,-self.max_len:,:]
        return self.k.value, self.v.value

def load_model(
    model_id: str, 
    model_cls: Type, 
    model_dir: str = None,
    config_cls: Type = Config,
    model_creator: Callable = None,
):
    repo_name, model_name = model_id.split('/')
    model_dir = download_model(repo_name, model_name, model_dir)
    config = load_config(model_dir, cls=config_cls)
    dtype = config.dtype
    if model_creator:
        graphdef, state = model_creator(config)
    else:
        graphdef, state = nnx.split(nnx.eval_shape(lambda: model_cls(config, rngs=nnx.Rngs(0))))
    state = dict(state.flat_state())
    for fpath in glob(f"{model_dir}/model*.safetensors"):
        for path, val in ((k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("mlp.weight", "mlp.kernel").replace("lm_head.weight", "lm_head.kernel").replace("embed_tokens.weight", "embed_tokens.embedding"), jnp.array(v, dtype=dtype).T if k.endswith('proj.weight') or k.endswith('mlp.weight') or k.endswith('lm_head.weight') else jnp.array(v, dtype=dtype)) for k, v in load_file(fpath).items()):
            path_tuple = tuple(int(part) if part.isdigit() else part for part in path.split('.'))
            if path_tuple in state:
                state[path_tuple].value = val
            else:
                print(f'{path_tuple} missing')
    model = nnx.merge(graphdef, nnx.State.from_flat_path(state))
    model.set_attributes(dtype=dtype, param_dtype=dtype)
    tokenizer = Tokenizer(repo_name='local', model_name=model_dir)
    return model, tokenizer, config


def generate(
    model_id: str,
    model_cls: Type,
    model_dir: str = 'models',
    config_cls: Type = Config,
    prompts = None, 
    max_new_tokens: int = 100,
    use_chat_template: bool = True,
    custom_tokenizer_fn: Callable = None,
    model_creator: Callable = None,
    stream = True,
    use_scan = False,
    use_jit = False,
    **kwargs
):
    if prompts is None:
        if use_chat_template:
            prompts = "Give me a short introduction to large language model."
        else:
            prompts = ["#write a quick sort algorithm\n", "#hello world in C\n"]
    model, tokenizer, config = load_model(model_id, model_cls, model_dir, config_cls, model_creator)
    if use_chat_template:
        assert isinstance(prompts, str)
        prompts = [{"role": "user", "content": prompts}]
        if 'add_generation_prompt' not in kwargs:
            kwargs['add_generation_prompt'] = True
    input_str, input_ids, position_ids, padding_mask = tokenizer(prompts, use_chat_template=use_chat_template, strftime_now=strftime_now, **kwargs)
    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    B, L = input_ids.shape
    position_ids = jnp.array(position_ids, dtype=jnp.float32)
    roper = Roper(config.head_dim, config.rope_theta, config.rope_traditional)
    total_len = max_new_tokens + L
    causal_mask = create_causal_mask(padding_mask).astype(config.dtype)
    causal_mask = jnp.pad(causal_mask, ((0,0), (0,0), (0,0), (max_new_tokens,0)), 'constant', constant_values=-1e4).astype(config.dtype) # new!
    generated_texts = [""] * B
    cache = [Cache(config.dtype, B, config.num_key_value_heads, total_len, config.head_dim) for _ in range(config.num_hidden_layers)]
    goon = jnp.ones((B, 1), dtype=bool)
    eos_id = config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0] # ad hoc
    start_tic = time.perf_counter()
    def scan_step(carry):
        input_ids, position_ids, causal_mask, cache, goon = carry
        rope = roper(position_ids)
        logits = model(input_ids, causal_mask, rope, cache)
        next_input_ids = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        next_input_ids = jnp.where(goon, next_input_ids, eos_id)
        new_mask = jnp.pad(causal_mask[:, :, -1:, 1:], ((0,0), (0,0), (0,0), (0,1)), 'constant', constant_values=0)
        goon = goon & (next_input_ids != eos_id)
        next_position_ids = position_ids[:, -1:] + 1
        new_carry = (next_input_ids, next_position_ids, new_mask, cache, goon)
        return new_carry, next_input_ids
    carry = (input_ids, position_ids, causal_mask, cache, jnp.ones((B, 1), dtype=bool))
    carry, first_tok = scan_step(carry)
    prompt_tic = time.perf_counter()
    if use_scan:
        scan_fn = nnx.scan(scan_step, in_axes=(nnx.Carry,), out_axes=(nnx.Carry, 1), length=max_new_tokens-1)
        carry, output_ids = scan_fn(carry)
        output_ids = jnp.concat([first_tok, jnp.squeeze(output_ids, -1)], axis=1).tolist()
    else:
        output_ids = [first_tok]
        if use_jit:
            scan_fn = nnx.jit(scan_step)
        else:
            scan_fn = scan_step
        for i in range(max_new_tokens-1):
            if stream:
                print(tokenizer.decode(output_ids[-1].tolist()[0]), end='', flush=True)
            carry, _output_ids = scan_fn(carry)
            output_ids.append(_output_ids)
            if not jnp.any(carry[-1]):
                break
        output_ids = jnp.concat(output_ids, axis=1).tolist()

    end_tic = time.perf_counter()
    for i, (i_str, o_ids) in enumerate(zip(input_str, output_ids)):
        o_ids = o_ids[:o_ids.index(eos_id)] if eos_id in o_ids else o_ids
        o_str = tokenizer.decode(o_ids)
        print(f'\n=== Input ===\n{i_str}\n=== Output===\n{o_str}\n')
    measure_performance(start_tic, prompt_tic, end_tic, B, L, max_new_tokens)
    return generated_texts, output_ids

