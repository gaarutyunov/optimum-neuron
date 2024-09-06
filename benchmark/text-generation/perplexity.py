# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


if importlib.util.find_spec("optimum.neuron") is not None:
    from optimum.neuron import NeuronModelForCausalLM


class Perplexity:
    """
    A class for calculating the perplexity of a language model.
    """

    def __init__(self, model, tokenizer, dataset_path="wikitext", dataset_name=None, split="test", text_column="text"):
        """
        Calculate perplexity using the same method as seen in llama.cpp.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The language model for which the perplexity is calculated.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the model.
        dataset_path : str, optional
            The path to the dataset on the Hugging Face dataset hub. Default is 'wikitext'.
        dataset_name : str, optional
            The name of the dataset. Default is None.
        split : str, optional
            The split of the dataset to use. Default is 'test'.
        text_column : str, optional
            The name of the column in the dataset that contains the text data. Default is 'text'.
        """
        self._model = model
        if hasattr(model.config, "neuron"):
            if not model.model.neuron_config.output_all_logits:
                raise ValueError(
                    "To evaluate the perplexity of a Neuron model, it must be configured to return all logits during export."
                )
            self._neuron = True
        else:
            self._neuron = False
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the dataset by loading and formatting.

        Returns
        -------
        str
            The formatted dataset as a single string.
        """
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"

        # Load the dataset
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        # Format the text column of the dataset
        text_list = [" \n" if s == "" else s for s in data[self._text_column]]
        return "".join(text_list)

    @staticmethod
    def softmax(logits):
        """
        Static method for applying the softmax function.

        Parameters
        ----------
        logits : np.ndarray
            The input to the softmax function.

        Returns
        -------
        np.ndarray
            The output of the softmax function.
        """
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def calculate_perplexity(self, n_ctx=512, n_batch=512):
        """
        Calculates the perplexity of the language model.

        Parameters
        ----------
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.

        Returns
        -------
        list
            The list of perplexity scores calculated.
        """
        # Tokenize the text
        self._tokenizer.model_max_length = sys.maxsize
        tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)

        nll = 0.0  # Negative log likelihood
        count = 0  # Counter for processed tokens
        curr_ppl = 0
        all_perplexity = []

        with tqdm(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                # Process each batch of tokens
                nll, count = self._process_batch(i, n_ctx, n_batch, tokens, nll, count)

                # Calculate and display the current perplexity
                curr_ppl = np.exp(nll / count)
                all_perplexity.append(curr_ppl)
                progress.set_description(f"Perplexity: {curr_ppl:.4f}")

        return all_perplexity

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        """
        Processes each batch of tokens.

        Parameters
        ----------
        i : int
            The batch index.
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.
        tokens : torch.Tensor
            The tokenized text.
        nll : float
            The current negative log likelihood.
        count : int
            The current count of processed tokens.

        Returns
        -------
        float
            The updated negative log likelihood.
        int
            The updated count of processed tokens.
        """
        start = i * n_ctx
        end = start + n_ctx

        num_batches = (n_ctx + n_batch - 1) // n_batch

        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)

            token_org = tokens[0][batch_start].item()

            if j == 0:
                # Replace the first token with the BOS token
                tokens[0][batch_start] = self._tokenizer.bos_token_id

            # Compute the logits for the current batch of tokens
            batch_logits = self._compute_batch_logits(tokens, batch_start, batch_size)

            tokens[0][batch_start] = token_org

            logits.append(batch_logits)

        # We rely on the fact that attention in the forward pass only looks at previous
        # tokens here, so the logits returned for each token are an accurate representation
        # of what the model would have predicted at that point.
        #
        # Example, we have a context window of 512, we will compute perplexity for each of the
        # last 256 tokens.  Then, we split the input up into context window size chunks to
        # process the entire prompt.

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j].cpu().numpy()
            # Compute the probability of the next token
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

            # Update the negative log likelihood and the count of processed tokens
            nll += -np.log(prob, where=prob > 0)
            count += 1

        return nll, count

    def _compute_batch_logits(self, tokens, batch_start, batch_size):
        """
        Computes the logits for a batch of tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            The tokenized text.
        batch_start : int
            The start index of the batch.
        batch_size : int
            The size of the batch.

        Returns
        -------
        torch.Tensor
            The logits for the batch of tokens.
        """
        input_ids = tokens[:, batch_start : batch_start + batch_size]
        # Compute the logits without keeping track of gradients
        with torch.no_grad():
            if self._neuron:
                attention_mask = torch.ones_like(input_ids)
                model_inputs = self._model.prepare_inputs_for_prefill(input_ids, attention_mask)
            else:
                model_inputs = {"input_ids": input_ids}
            outputs = self._model.forward(**model_inputs)
        return outputs.logits.detach()


def perplexity(
    model,
    tokenizer,
    stride: int = 512,
):
    print("Evaluating perplexity")
    ppl = Perplexity(model, tokenizer)
    return np.mean(ppl.calculate_perplexity(n_ctx=stride))


def main():
    parser = argparse.ArgumentParser(description="Evaluate neuron model perplexity")
    parser.add_argument("model", type=str, help="The path to the model to evaluate.")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_path = args.model
    config = AutoConfig.from_pretrained(model_path)
    model_name = config._name_or_path
    if hasattr(config, "neuron"):
        model_type = "neuron"
        model = NeuronModelForCausalLM.from_pretrained(model_path)
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model_type = "torch"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", low_cpu_mem_usage=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"{model_name}: evaluating perplexity of {model_type} model")
    ppl = perplexity(model, tokenizer)
    print(ppl)


if __name__ == "__main__":
    main()
