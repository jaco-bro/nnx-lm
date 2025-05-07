from setuptools import find_packages, setup

requirements = [
    "jax-metal==0.1.1; platform_system=='Darwin'",
    "jaxlib==0.5.0; platform_system=='Darwin'",
    "jax==0.5.0; platform_system=='Darwin'",
    "flax==0.10.4",
    # "safetensors>=0.5.3",
    # "pydantic>=2.11.3",
    # "huggingface-hub>=0.30.2",
    "safetensors",
    "pydantic",
    "huggingface-hub",
    "tokenizerz==0.0.2a7",
]

extras_require = {
    'gpu': ['jax[cuda12]'],
    'tpu': ['jax[tpu]'],
}

setup(
    name='nnx-lm',
    url='https://github.com/jaco-bro/nnx-lm',
    packages=find_packages(),
    version='0.0.1',
    readme="README.md",
    author_email="backupjjoe@gmail.com",
    description="nnx-lm: A portable, pip-installable CLI for running LLMs via JAX on any hardware backend.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="J Joe",
    license="Apache-2.0",
    # python_requires=">=3.12.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "nlm=nnxlm.qwen3:main",
        ],
    },
)
