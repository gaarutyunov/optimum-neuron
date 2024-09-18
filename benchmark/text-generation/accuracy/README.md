# Neuron LLM models accuracy

EleutherAI [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is compatible with neuron models
exported using `optimum-neuron`.

## Prerequisite

Install the harness from source following the instructions in the harness [repository](https://github.com/EleutherAI/lm-evaluation-harness).

Note: the current upstream version is not very flexible, so you should use [this branch](https://github.com/dacorvo/lm-evaluation-harness/tree/update_neuron) instead until this [pull-request](https://github.com/EleutherAI/lm-evaluation-harness/pull/2314) is merged.

## Evaluate a neuron model

The evaluation of a neuron model is as simple as:

```shell
lm_eval --model neuronx \
        --tasks gsm8k \
        --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tp_degree=24,dtype=bfloat16 \
        --batch_size 8
```
The model will be exported before evaluation with the specified parameters.

Note that if you are using the aforementioned branch you can also:

- set the max_length,
- evaluate a pre-exported neuron model (either from the hub or local).
