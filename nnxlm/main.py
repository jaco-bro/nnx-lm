from tokenizerz import Tokenizer
import argparse
from .utils import generate
from .qwen3 import Qwen3ForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .glm4 import Glm4ForCausalLM
from .granite import GraniteForCausalLM
from .llama import LlamaForCausalLM
from .utils import load_config, download_repo, load_model, generate

ARCH_MAPPING = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "GraniteForCausalLM": GraniteForCausalLM,
    "Glm4ForCausalLM": Glm4ForCausalLM,
}

def load(model_id, model_dir='models'):
    repo_name, model_name = model_id.split('/')
    model_dir = download_repo(repo_name, model_name, model_dir)
    config = load_config(model_dir)
    model, tokenizer = load_model(model_dir, config=config, cls= ARCH_MAPPING.get(config.architectures[0]))
    return model, tokenizer, config

def test(prompts=None, max_new_tokens=3, use_scan=False, use_jit=False):
    for model_id in ['Qwen/Qwen3-0.6B','THUDM/GLM-4-9B-0414', 'ibm-granite/granite-3.3-2b-instruct']:
        model, tokenizer, config = load(model_id)
        generate(model, tokenizer, config,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    use_chat_template=True,
                    use_scan=use_scan,
                    use_jit=use_jit,
                 )
    for model_id in ['Qwen/Qwen2.5-Coder-0.5B', 'HuggingFaceTB/SmolLM2-135M']:
        model, tokenizer, config = load(model_id)
        generate(model, tokenizer, config,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    use_chat_template=False,
                    use_scan=use_scan,
                    use_jit=use_jit,
                 )

def cli():
    parser = argparse.ArgumentParser(description="Load a model and generate text.")
    parser.add_argument("--model", type=str, default='Qwen/Qwen3-0.6B', dest="model_id", help="Model ID in the format 'repo/model_name'.")
    parser.add_argument("--prompts", type=str, nargs='*', help="Prompt(s) for generation.")
    parser.add_argument("--max", type=int, default=100, help="Maximum new tokens to generate.")
    parser.add_argument("--format", dest="use_chat_template", action="store_true", help="Use chat template formatting.")
    parser.add_argument("--no-format", dest="use_chat_template", action="store_false", help="Do not use chat template.")
    parser.set_defaults(use_chat_template=True)
    parser.add_argument("--stream", dest="stream", action="store_true", help="Stream the output.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Do not stream output.")
    parser.set_defaults(stream=True)
    parser.add_argument("--scan", action="store_true", help="Enable scan mode.")
    parser.add_argument("--jit", action="store_true", help="Enable JIT compilation.")
    parser.add_argument("--dir", type=str, default="models", help="Directory to download/load models.")
    args =  parser.parse_args()
    model, tokenizer, config = load(args.model_id, model_dir=args.dir)
    s, i = generate(
        model,
        tokenizer,
        config,
        prompts=args.prompts,
        max_new_tokens=args.max,
        use_chat_template=args.use_chat_template,
        stream=args.stream,
        use_scan=args.scan,
        use_jit=args.jit
    )
    # for n, (_s, _i) in enumerate(zip(s, i)):
    #     print('=== {n} ===')
    #     print(_s)
    #     print(_i)

if __name__ == "__main__":
    # cli()
    test()
