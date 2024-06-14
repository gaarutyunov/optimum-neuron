#!/bin/bash

endpoint_name=${1}
vu=${2:-1}

benchmark_script=${LLMPerf}/token_benchmark_ray.py

if ! test -f ${benchmark_script}; then
  echo "LLMPerf script not found, please export LLMPerf=<path-to-llmperf>."
fi

max_requests=$(expr ${vu} \* 8 )
date_str=$(date '+%Y-%m-%d-%H-%M-%S')

MESSAGES_API=true python ${benchmark_script} \
       --model ${endpoint_name} \
       --llm-api "sagemaker" \
       --mean-input-tokens 1500 \
       --stddev-input-tokens 150 \
       --mean-output-tokens 245 \
       --stddev-output-tokens 20 \
       --max-num-completed-requests ${max_requests} \
       --timeout 7200 \
       --num-concurrent-requests ${vu} \
       --results-dir "sagemaker_bench_results/${date_str}"
