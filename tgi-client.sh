#!/bin/sh

repeat=${1:-1}
max_new_tokens=${2:-128}

for i in $(seq ${repeat}); do
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"One of my fondest memory is","parameters":{"do_sample": true, "top_k": 50, "max_new_tokens":'${max_new_tokens}', "return_full_text": true}}' \
    -H 'Content-Type: application/json'
    echo "\nRequest ${i} completed."
done
