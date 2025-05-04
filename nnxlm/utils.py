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
from safetensors.numpy import load_file
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

def get_model_files(repo, model):
    base_url = f"https://huggingface.co/{repo}/{model}/resolve/main"
    model_dir = model
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
        print("Falling back to default file list.")
        files.append("model.safetensors")
    return [(f"{base_url}/{file}", os.path.join(model_dir, file), file) for file in files]

def download_model(repo, model, max_workers=4):
    tasks = get_model_files(repo, model)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, path, desc) for url, path, desc in tasks]
        for future in futures:
            future.result()

@dataclass
class Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
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

def load_config(model_name, cls=Config):
    with open(f'{model_name}/config.json', 'r') as f:
        config_dict = json.load(f)
    return cls(**{k: v for k, v in config_dict.items() if k in {f.name for f in fields(cls)}})


class Roper:
    def __init__(self, head_dim, theta=10000.0, traditional=True):
        dim = head_dim // 2
        self.freq = 1.0 / (theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))

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
    return q_out, k_out


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
    return q_out, k_out

@jax.jit
def create_causal_mask(padding_mask):
    padding_mask = jnp.array(padding_mask)
    seq_length = padding_mask.shape[1]
    causal_matrix = jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))
    causal_mask = jnp.where(causal_matrix & padding_mask[:, None, :], 0.0, -1e10)
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
    print(f"Prompt processing: {prompt_throughput:.1f} tokens/sec ({tokens_processed} tokens in {prompt_duration:.1f}s)")
    print(f"Token generation: {generation_throughput:.1f} tokens/sec ({tokens_generated} tokens in {generation_duration:.1f}s)")
    return metrics

def load_model(
    model_id: str, 
    model_cls: Type, 
    config_cls: Type = Config,
    model_creator: Callable = None
):
    repo_name, model_name = model_id.split('/')
    download_model(repo_name, model_name)
    config = load_config(model_name, cls=config_cls)
    if model_creator:
        graphdef, state = model_creator(config)
    else:
        graphdef, state = nnx.split(nnx.eval_shape(lambda: model_cls(config, rngs=nnx.Rngs(0))))
    state = dict(state.flat_state())
    for fpath in glob(f"{model_name}/model*.safetensors"):
        for path, val in ((k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("mlp.weight", "mlp.kernel").replace("lm_head.weight", "lm_head.kernel").replace("embed_tokens.weight", "embed_tokens.embedding"), nnx.Param(jnp.array(v).T) if k.endswith('proj.weight') or k.endswith('mlp.weight') or k.endswith('lm_head.weight') else nnx.Param(jnp.array(v))) for k, v in load_file(fpath).items()):
            path_tuple = tuple(int(part) if part.isdigit() else part for part in path.split('.'))
            if path_tuple in state:
                state[path_tuple].value = val
            else:
                print(f'{path_tuple} missing')
    model = nnx.merge(graphdef, nnx.State.from_flat_path(state))
    dtype = eval(f'jnp.{config.torch_dtype}')
    model.set_attributes(dtype=dtype, param_dtype=dtype)
    tokenizer = Tokenizer(repo_name=repo_name, model_name=model_name)
    return model, tokenizer, config

def generate(
    model_id: str,
    model_cls: Type,
    config_cls: Type = Config,
    prompts = None, 
    max_new_tokens: int = 5,
    use_chat_template: bool = True,
    custom_tokenizer_fn: Callable = None,
    model_creator: Callable = None
):
    if prompts is None:
        if use_chat_template:
            prompts = "Give me a short introduction to large language model."
        else:
            prompts = ["#write a quick sort algorithm", "#hello world"]
    model, tokenizer, config = load_model(model_id, model_cls, config_cls, model_creator)
    if use_chat_template:
        assert isinstance(prompts, str)
        prompts = [{"role": "user", "content": prompts}]
    input_ids, position_ids, padding_mask = tokenizer(prompts, use_chat_template=use_chat_template, strftime_now=strftime_now)
    causal_mask = create_causal_mask(padding_mask)
    input_ids = jnp.array(input_ids, dtype=jnp.int32)
    B, L = input_ids.shape
    position_ids = jnp.array(position_ids, dtype=jnp.float32)
    roper = Roper(config.head_dim, config.rope_theta, config.rope_traditional)
    output_ids = []
    generated_texts = [""] * B
    cache = None
    start_tic = time.perf_counter()
    for i in range(max_new_tokens):
        rope = roper(position_ids)
        logits, cache = model(input_ids, causal_mask, rope, cache)
        if i == 0:
            prompt_tic = time.perf_counter()
        input_ids = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        new_tokens = input_ids.tolist()
        print(f'{new_tokens=}')
        for b, token_id in enumerate(new_tokens):
            token_text = tokenizer.decode(token_id)
            generated_texts[b] += token_text
            print(f"Batch {b}, Token {i+1}: {token_text!r} -> {generated_texts[b]!r}")
        output_ids.append(input_ids)
        position_ids = position_ids[:, -1:] + 1
        causal_mask = jnp.pad(causal_mask[:, :, -1:, :], ((0, 0), (0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)
    end_tic = time.perf_counter()
    measure_performance(start_tic, prompt_tic, end_tic, B, L, max_new_tokens)
    output_ids = jnp.concatenate(output_ids, axis=1).tolist()
    output_str = [tokenizer.decode(o) for o in output_ids]
    print(f'{output_ids=}')
    print(f'{output_str=}')
    return generated_texts, output_ids

