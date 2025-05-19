from datetime import datetime
import os
import functools
import json
import time
import math
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
from pathlib import Path

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

def download_repo(repo, model, dest=None, max_workers=4):
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
    partial_rotary_factor: float = 1.0
    max_position_embeddings: Optional[int] = None
    original_max_position_embeddings: Optional[int] = None
    logits_scaling: float = 1.0
    attention_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def dtype(self):
        return eval(f'jnp.{self.torch_dtype}')

def get_nested(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

def load_config(model_name, cls=Config):
    with open(Path(model_name) / 'config.json', 'r') as f:
        config_dict = json.load(f)
    cls_fields = {f.name for f in fields(cls)}
    init_args = {k: v for k, v in config_dict.items() if k in cls_fields}
    extra_args = {}
    for k, v in config_dict.items():
        if k not in cls_fields:
            extra_args[k] = v
    return cls(**init_args, extra_config=extra_args)

def load_model(model_dir, config, cls, model_creator=None):
    dtype = config.dtype
    if model_creator:
        graphdef, state = model_creator(config)
    else:
        graphdef, state = nnx.split(nnx.eval_shape(lambda: cls(config, rngs=nnx.Rngs(0))))
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
    return model, tokenizer

class Roper(nnx.Module):
    def __init__(self, config, su_len):
        self.su_scale = 1.0
        if get_nested(config.extra_config, ["rope_scaling", "rope_type"])=='llama3':
            self._llama3(config)
        elif get_nested(config.extra_config, ["rope_scaling", "type"])=='longrope':
            self._su(config, su_len)
        else:
            dim = int(config.head_dim*config.partial_rotary_factor/2)
            self.freq = nnx.Variable(1.0 / (config.rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim)))

    @nnx.jit
    def __call__(self, positions):
        positions = positions[:, None, :, None]
        angles = positions * self.freq
        cos = jnp.cos(angles) * self.su_scale
        sin = jnp.sin(angles) * self.su_scale
        return jax.lax.stop_gradient(cos), jax.lax.stop_gradient(sin)
    
    def _llama3(self, config):
        rot_dims = int(config.head_dim * config.partial_rotary_factor)
        scaling_config = get_nested(config.extra_config, ["rope_scaling"])
        factor = scaling_config.get("factor", 1.0)
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_max = scaling_config.get("original_max_position_embeddings", 8192)
        idx = jnp.arange(0, rot_dims, 2, dtype=jnp.float32)
        freqs = config.rope_theta ** (idx / rot_dims)
        wavelens = 2 * jnp.pi * freqs
        low_wl = old_max / low_freq_factor
        high_wl = old_max / high_freq_factor
        freqs_adj = jnp.where(wavelens > low_wl, freqs * factor, freqs)
        is_med = (wavelens > high_wl) & (wavelens < low_wl)
        smooth = (old_max / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_freqs = freqs_adj / ((1 - smooth) / factor + smooth)
        freqs_final = jnp.where(is_med, smooth_freqs, freqs_adj)
        self.freq = nnx.Variable(1.0 / freqs_final)

    def _su(self, config, su_len):
        factor = 'long' if su_len > config.original_max_position_embeddings else 'short'
        self.su_scale = math.sqrt(1.0 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        rot_dims = int(config.head_dim * config.partial_rotary_factor)
        scaling_config = get_nested(config.extra_config, ["rope_scaling"])
        freqs = config.rope_theta ** (jnp.arange(0, rot_dims, 2, dtype=jnp.float32) / rot_dims)
        factor = scaling_config.get(f'{factor}_factor')
        factor = jnp.array(factor, dtype=jnp.float32)
        self.freq = nnx.Variable(1.0 / (freqs * factor))

@functools.partial(jax.jit, static_argnames=['rot_dims', 'traditional'])
def apply_rope(q, k, cos, sin, rot_dims=None, traditional=False):
    if rot_dims is None:
        q_rot, k_rot = q, k
    else:
        q_rot, q_pass = q[..., :rot_dims], q[..., rot_dims:]
        k_rot, k_pass = k[..., :rot_dims], k[..., rot_dims:]
    if traditional:
        q_even = q_rot[..., 0::2]
        q_odd  = q_rot[..., 1::2]
        k_even = k_rot[..., 0::2]
        k_odd  = k_rot[..., 1::2]
        q_rotated = jnp.stack([(q_even * cos - q_odd * sin), (q_even * sin + q_odd * cos)], axis=-1).reshape(q_rot.shape).astype(q.dtype)
        k_rotated = jnp.stack([(k_even * cos - k_odd * sin), (k_even * sin + k_odd * cos)], axis=-1).reshape(k_rot.shape).astype(k.dtype)
    else:
        q_split = q_rot.reshape(*q.shape[:-1], 2, -1)
        k_split = k_rot.reshape(*k.shape[:-1], 2, -1)
        q_rotated = jnp.concatenate([
            q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
            q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
        ], axis=-1).astype(q.dtype)
        k_rotated = jnp.concatenate([
            k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
            k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
        ], axis=-1).astype(k.dtype)
    if rot_dims is None:
        return q_rotated, k_rotated
    else:
        q_out = jnp.concatenate([q_rotated, q_pass], axis=-1)
        k_out = jnp.concatenate([k_rotated, k_pass], axis=-1)
        return q_out, k_out

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

def generate(
    model,
    tokenizer,
    config,
    prompts, 
    max_new_tokens: int = 100,
    use_chat_template: bool = True,
    custom_tokenizer_fn: Callable = None,
    model_creator: Callable = None,
    stream = True,
    use_scan = False,
    use_jit = False,
    **kwargs
):
    if isinstance(prompts, str):
        prompts = [prompts]
    if use_chat_template:
        try:
            if 'add_generation_prompt' not in kwargs:
                kwargs['add_generation_prompt'] = True
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], strftime_now=strftime_now, **kwargs) for prompt in prompts]
        except Exception as e:
            print(e)
    # input_str, input_ids, position_ids, padding_mask = tokenizer(prompts, use_chat_template=use_chat_template, strftime_now=strftime_now, **kwargs)
    input_str, input_ids, position_ids, padding_mask = tokenizer(prompts)
    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    B, L = input_ids.shape
    position_ids = jnp.array(position_ids, dtype=jnp.float32)
    total_len = max_new_tokens + L
    roper = Roper(config, total_len)
    causal_mask = create_causal_mask(padding_mask).astype(config.dtype)
    causal_mask = jnp.pad(causal_mask, ((0,0), (0,0), (0,0), (max_new_tokens,0)), 'constant', constant_values=-1e4).astype(config.dtype)
    cache = [Cache(config.dtype, B, config.num_key_value_heads, total_len, config.head_dim) for _ in range(config.num_hidden_layers)]
    zeropad = jnp.zeros((B, 1, 1, 1), dtype=config.dtype)
    goon = jnp.ones((B, 1), dtype=bool)
    eos_id = config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0] # ad hoc
    start_tic = time.perf_counter()
    def scan_step(carry):
        input_ids, position_ids, causal_mask, cache, goon = carry
        rope = roper(position_ids)
        logits = model(input_ids, causal_mask, rope, cache)
        next_input_ids = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        next_input_ids = jnp.where(goon, next_input_ids, eos_id)
        new_mask = jnp.concat([causal_mask[:, :, -1:, :], zeropad], axis=-1)[:,:,:,1:]
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
            scan_fn = nnx.jit(scan_step, donate_argnums=(0,)) # : Donation is not implemented for ('METAL',).
        else:
            scan_fn = scan_step
        for i in range(max_new_tokens-1):
            carry, _output_ids = scan_fn(carry)
            if stream:
                print(tokenizer.decode(output_ids[-1].tolist()[0]), end='', flush=True)
            output_ids.append(_output_ids)
            if not jnp.any(carry[-1]):
                break
        output_ids = jnp.concat(output_ids, axis=1).tolist()

    end_tic = time.perf_counter()
    output_str = []
    for i, (i_str, o_ids) in enumerate(zip(input_str, output_ids)):
        o_ids = o_ids[:o_ids.index(eos_id)] if eos_id in o_ids else o_ids
        o_str = tokenizer.decode(o_ids)
        output_str.append(o_str)
        print(f'\n=== Input ===\n{i_str}\n=== Output===\n{o_str}\n')
    measure_performance(start_tic, prompt_tic, end_tic, B, L, max_new_tokens)
    return output_str, output_ids
