#!/bin/sh

repeat=${1:-1}
max_new_tokens=${2:-128}

for i in $(seq ${repeat}); do
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"do_sample": true, "top_k": 50, "max_new_tokens":'${max_new_tokens}'}}' \
    -H 'Content-Type: application/json'
    echo "\nRequest ${i} completed."
done
