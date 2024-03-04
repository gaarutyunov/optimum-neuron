#!/bin/sh
set -e

model=${1:-NousResearch/Llama-2-7b-chat-hf}
batch_size=${2:-4}
seq_length=${3:-4096}

max_input_length=$((${seq_length} / 2))
max_total_tokens=${seq_length}
max_batch_prefill_tokens=$(( ${batch_size} * ${max_input_length}))
max_batch_total_tokens=$(( ${batch_size} * ${seq_length}))

docker run --rm -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       --device=/dev/neuron1 \
       --device=/dev/neuron2 \
       --device=/dev/neuron3 \
       --device=/dev/neuron4 \
       --device=/dev/neuron5 \
       --device=/dev/neuron6 \
       --device=/dev/neuron7 \
       --device=/dev/neuron8 \
       --device=/dev/neuron9 \
       --device=/dev/neuron10 \
       --device=/dev/neuron11 \
       neuronx-tgi:latest \
       --model-id $model \
       --max-batch-size ${batch_size} \
       --max-concurrent-requests 128 \
       --max-input-length ${max_input_length} \
       --max-total-tokens ${max_total_tokens}
