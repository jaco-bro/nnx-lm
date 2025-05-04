import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from .utils import (
    apply_partial_rope,
    generate,
)

class Glm4MLP(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.gate_up_proj = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=2 * config.intermediate_size, 
            use_bias=False, 
            rngs=rngs
        )
        self.down_proj = nnx.Linear(
            in_features=config.intermediate_size, 
            out_features=config.hidden_size, 
            use_bias=False, 
            rngs=rngs
        )
    
    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.gate_up_proj(x)
        gate, up_states = jnp.split(x, 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up_states)

class Glm4Attention(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = self.head_dim ** -0.5
        self.partial_rotary_factor = config.partial_rotary_factor
        self.q_proj = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=self.n_heads * self.head_dim, 
            use_bias=config.attention_bias, 
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=self.n_kv_heads * self.head_dim, 
            use_bias=config.attention_bias, 
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=self.n_kv_heads * self.head_dim, 
            use_bias=config.attention_bias, 
            rngs=rngs
        )
        self.o_proj = nnx.Linear(
            in_features=self.n_heads * self.head_dim, 
            out_features=config.hidden_size, 
            use_bias=False, 
            rngs=rngs
        )
        self.rope_theta = config.rope_theta
    
    @nnx.jit
    def __call__(self, x, attention_mask, rope, cache=None):
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim)
        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))
        queries, keys = apply_partial_rope(queries, keys, *rope, self.partial_rotary_factor)
        if cache is not None:
            keys = jnp.concatenate([cache[0], keys], axis=2) 
            values = jnp.concatenate([cache[1], values], axis=2)
        cache = (keys, values)
        if self.n_heads > self.n_kv_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            keys = jnp.repeat(keys, repeats=repeat_factor, axis=1)
            values = jnp.repeat(values, repeats=repeat_factor, axis=1)
        attn_scores = jnp.matmul(queries, jnp.swapaxes(keys, -1, -2)) * self.scale
        attn_scores = attn_scores + attention_mask
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, values)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(B, L, -1)
        return self.o_proj(attn_output), cache

class Glm4DecoderLayer(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.self_attn = Glm4Attention(config, rngs=rngs)
        self.mlp = Glm4MLP(config, rngs=rngs)
        self.input_layernorm = nnx.RMSNorm(
            num_features=config.hidden_size, 
            epsilon=config.rms_norm_eps, 
            rngs=rngs
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            num_features=config.hidden_size, 
            epsilon=config.rms_norm_eps, 
            rngs=rngs
        )
        self.post_self_attn_layernorm = nnx.RMSNorm(
            num_features=config.hidden_size, 
            epsilon=config.rms_norm_eps, 
            rngs=rngs
        )
        self.post_mlp_layernorm = nnx.RMSNorm(
            num_features=config.hidden_size, 
            epsilon=config.rms_norm_eps, 
            rngs=rngs
        )
    
    @nnx.jit
    def __call__(self, x, attention_mask, rope, cache=None):
        attn_output, c = self.self_attn(
            self.input_layernorm(x),
            attention_mask=attention_mask,
            rope=rope,
            cache=cache,
        )
        x = x + self.post_self_attn_layernorm(attn_output)
        residual = x
        mlp_output = self.mlp(self.post_attention_layernorm(x))
        x = residual + self.post_mlp_layernorm(mlp_output)
        return x, c

class Glm4Model(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )
        self.layers = [Glm4DecoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)]
        self.norm = nnx.RMSNorm(
            num_features=config.hidden_size, 
            epsilon=config.rms_norm_eps, 
            rngs=rngs
        )
    
    def __call__(self, input_ids, attention_mask, rope, cache=None):
        x = self.embed_tokens(input_ids)
        c = []
        for i, layer in enumerate(self.layers):
            x, c_ = layer(
                x,
                attention_mask=attention_mask,
                rope=rope,
                cache=cache[i] if cache else None,
            )
            c.append(c_)
        return self.norm(x), c

class Glm4ForCausalLM(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.model = Glm4Model(config, rngs=rngs)
        self.lm_head = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=config.vocab_size, 
            use_bias=False, 
            rngs=rngs
        )
    
    def __call__(self, input_ids, attention_mask, rope, cache=None):
        x, c = self.model(input_ids, attention_mask=attention_mask, rope=rope, cache=cache)
        return self.lm_head(x), c

if __name__ == "__main__":
    generate(model_id='THUDM/GLM-4-9B-0414', model_cls=Glm4ForCausalLM)
