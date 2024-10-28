from models.llama.neuron_modeling_llama import NeuronLlamaModel

from optimum.exporters.base import ExportConfig


_EXPORT_CONFIGS = {}


def get_export_config(model_type: str):
    return _EXPORT_CONFIGS[model_type]


def register_export_config(model_type):
    def wrapper(cls):
        _EXPORT_CONFIGS[model_type] = cls
        return cls

    return wrapper


@register_export_config("llama")
class LlamaNeuronExportConfig(ExportConfig):

    _STATE_DICT_MODEL_PREFIX = "model."
    _MODEL_CLS = NeuronLlamaModel
    _ATTN_CLS = "NeuronLlamaAttention"

    @staticmethod
    def get_compiler_args():
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' -O1 "
