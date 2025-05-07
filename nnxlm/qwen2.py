import jax
import jax.numpy as jnp
from flax import nnx

from .utils import (
# from utils import (
    apply_rope,
    generate
)

class Attention(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.n_repeat = self.num_attention_heads // self.num_kv_heads
        self.q_proj = nnx.Linear(in_features=config.hidden_size, out_features=self.num_attention_heads * self.head_dim, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(in_features=config.hidden_size, out_features=self.num_kv_heads * self.head_dim, use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(in_features=config.hidden_size, out_features=self.num_kv_heads * self.head_dim, use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(in_features=self.num_attention_heads * self.head_dim, out_features=config.hidden_size, use_bias=False, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x, attention_mask, rope, cache):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        q, k = apply_rope(q, k, *rope)
        k, v = cache(k, v)
        if self.n_repeat > 1:
            k = k.repeat(repeats=self.n_repeat, axis=1)
            v = v.repeat(repeats=self.n_repeat, axis=1)
        w = jnp.matmul(q, k.swapaxes(-1, -2)) * self.scale
        w = w + attention_mask
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.matmul(w, v)
        w = w.transpose((0, 2, 1, 3))
        w = w.reshape(B, L, -1)
        return self.o_proj(w)

class MLP(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, use_bias=False, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x: jax.Array):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(jax.nn.silu(gate) * up)

class TransformerBlock(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.self_attn = Attention(config, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)
        self.input_layernorm = nnx.RMSNorm(num_features=config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = nnx.RMSNorm(num_features=config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x, attention_mask, rope, cache):
        h = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, rope=rope, cache=cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))

class Qwen2Model(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.layers = [TransformerBlock(config, rngs=rngs) for _ in range(config.num_hidden_layers)]
        self.embed_tokens = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs)
        self.norm = nnx.RMSNorm(num_features=config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)
    
    def __call__(self, input_ids, attention_mask, rope, cache):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attention_mask, rope=rope, cache=cache[i])
        return self.norm(x)
    
class Qwen2ForCausalLM(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.tie = tie = config.tie_word_embeddings
        self.model = Qwen2Model(config, rngs=rngs)
        if not tie:
            self.lm_head = nnx.Linear(in_features=config.hidden_size, out_features=config.vocab_size, use_bias=False, rngs=rngs)
    
    def __call__(self, input_ids, attention_mask, rope, cache):
        x = self.model(input_ids, attention_mask=attention_mask, rope=rope, cache=cache)
        if self.tie:
            x = self.model.embed_tokens.attend(x)
        else:
            x = self.lm_head(x)
        return x

if __name__ == "__main__":
    generate(model_id='Qwen/Qwen2.5-Coder-0.5B', model_cls=Qwen2ForCausalLM, use_chat_template=False)
