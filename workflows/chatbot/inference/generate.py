import argparse
import copy, time
import torch
import torch.nn.functional as F
import re, os, logging
from threading import Thread
from peft import PeftModel
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    StoppingCriteriaList,
    StoppingCriteria,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROMPT_DICT = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model_path", type=str, default="")
    parser.add_argument("-pm", "--peft_model_path", type=str, default="")
    parser.add_argument(
        "-ins",
        "--instructions",
        type=str,
        nargs="+",
        default=[
            "Tell me about alpacas.",
            "Tell me five words that rhyme with 'shock'.",
        ],
    )
    # Add arguments for temperature, top_p, top_k and repetition_penalty
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="The value used to control the randomness of sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="The cumulative probability of tokens to keep for sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="The number of highest probability tokens to keep for sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams for beam search.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The penalty applied to repeated tokens.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="specify tokenizer name"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="enable when use custom model architecture that is not yet part of the Hugging Face transformers package like MPT",
    )

    # habana parameters
    parser.add_argument(
        "--habana",
        action="store_true",
        help="Whether run on habana",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Whether to use jit trace. It should speed up generation.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    args = parser.parse_args()
    return args


class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: list[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False


def max_input_len(model, outlen=0):
    # need to adjust due to perf and real usage
    if hasattr(model.config, "max_seq_len"):
        return max((model.config.max_seq_len >> 2) - outlen, 0)
    if hasattr(model.config, "max_position_embeddings"):
        return max((model.config.max_position_embeddings >> 2) - outlen, 0)

    return 0


def create_prompts(examples):
    prompts = []
    for example in examples:
        prompt_template = (
            PROMPT_DICT["prompt_with_input"]
            if example["input"] != ""
            else PROMPT_DICT["prompt_without_input"]
        )
        prompt = prompt_template.format_map(example)
        prompts.append(prompt)
    return prompts


def get_optimized_model_name(config):
    from optimum.habana.transformers.generation import (
        MODELS_OPTIMIZED_WITH_STATIC_SHAPES,
    )

    for model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
        if model_type == config.model_type:
            return model_type

    return None


def get_ds_injection_policy(config):
    model_type = get_optimized_model_name(config)
    policy = {}
    if model_type:
        if model_type == "bloom":
            from transformers.models.bloom.modeling_bloom import BloomBlock

            policy = {BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}

        if model_type == "opt":
            from transformers.models.opt.modeling_opt import OPTDecoderLayer

            policy = {OPTDecoderLayer: ("self_attn.out_proj", ".fc2")}

        if model_type == "gpt2":
            from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

            policy = {GPT2MLP: ("attn.c_proj", "mlp.c_proj")}

        if model_type == "gptj":
            from transformers.models.gptj.modeling_gptj import GPTJBlock

            policy = {GPTJBlock: ("attn.out_proj", "mlp.fc_out")}

        if model_type == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

            policy = {GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")}

    return policy


MODELS = {}


def load_model(
    model_name,
    tokenizer_name,
    device="cpu",
    use_hpu_graphs=False,
    cpu_jit=False,
    use_cache=False,
):
    """
    Load the model and initialize the tokenizer.

    Args:
        model_name (str): The name of the model.
        device (str, optional): The device for the model. Defaults to 'cpu'. The valid value is 'cpu' or 'hpu'.
        use_hpu_graphs (bool, optional): Whether to use HPU graphs. Defaults to False. Only set when device is hpu.

    Returns:
        None

    Raises:
        ValueError: If the model is not supported, ValueError is raised.
    """
    print("Loading model {}".format(model_name))
    MODELS[model_name] = {}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False if re.search("llama", model_name, re.IGNORECASE) else True,
    )
    if re.search("flan-t5", model_name, re.IGNORECASE):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )
    elif re.search("mpt", model_name, re.IGNORECASE):
        from models.mpt.modeling_mpt import MPTForCausalLM

        model = MPTForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            torchscript=cpu_jit,
        )
    elif (
        re.search("gpt", model_name, re.IGNORECASE)
        or re.search("bloom", model_name, re.IGNORECASE)
        or re.search("llama", model_name, re.IGNORECASE)
        or re.search("opt", model_name, re.IGNORECASE)
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    else:
        raise ValueError(
            f"Unsupported model {model_name}, only supports FLAN-T5/LLAMA/MPT/GPT/BLOOM/OPT now."
        )

    if re.search("llama", model.config.architectures[0], re.IGNORECASE):
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2

    if (
        hasattr(model.generation_config, "pad_token_id")
        and model.generation_config.pad_token_id is not None
    ):
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    if (
        hasattr(model.generation_config, "eos_token_id")
        and model.generation_config.eos_token_id is not None
    ):
        tokenizer.eos_token_id = model.generation_config.eos_token_id
    if (
        hasattr(model.generation_config, "bos_token_id")
        and model.generation_config.bos_token_id is not None
    ):
        tokenizer.bos_token_id = model.generation_config.bos_token_id

    if tokenizer.pad_token_id is None:
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        ) = tokenizer.eos_token_id

    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if device == "hpu":
        model = model.eval().to("hpu")

        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model)
    else:
        import intel_extension_for_pytorch as intel_ipex

        model = intel_ipex.optimize(
            model.eval(),
            dtype=torch.bfloat16,
            inplace=True,
            level="O1",
            auto_kernel_selection=True,
        )
        if cpu_jit and re.search("mpt-7b", model_name, re.IGNORECASE):
            from models.mpt.mpt_trace import jit_trace_mpt_7b, MPTTSModelForCausalLM

            model = jit_trace_mpt_7b(model)
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = MPTTSModelForCausalLM(
                model, config, use_cache=use_cache, model_dtype=torch.bfloat16
            )

    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    MODELS[model_name]["model"] = model
    MODELS[model_name]["tokenizer"] = tokenizer
    print("model loaded")


def predict_stream(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "mosaicml/mpt-7b-chat".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 1
    model_name = (
        params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_kv_cache"] if "use_kv_cache" in params else False
    prompt = params["prompt"]

    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    if device == "cpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        stop_token_ids = [model.generation_config.eos_token_id]
        stop_token_ids.append(tokenizer(".", return_tensors="pt").input_ids)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )

        def generate_output():
            with torch.no_grad():
                with torch.cpu.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
                ):
                    generation_kwargs = dict(
                        streamer=streamer,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                    )
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [
                            StopOnTokens(
                                min_length=max(max_new_tokens - 20, 0),
                                start_length=input_token_len,
                                stop_token_id=stop_token_ids,
                            )
                        ]
                    )
                    return model.generate(**input_tokens, **generation_kwargs)

        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    elif device == "hpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_input_len(model, max_new_tokens),
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        stop_token_ids = [model.generation_config.eos_token_id]
        stop_token_ids.append(tokenizer(".", return_tensors="pt").input_ids)
        generate_kwargs = {
            "stopping_criteria": StoppingCriteriaList(
                [
                    StopOnTokens(
                        min_length=max(max_new_tokens - 20, 0),
                        start_length=input_token_len,
                        stop_token_id=stop_token_ids,
                    )
                ]
            )
        }
        is_graph_optimized = False
        if (
            re.search("gpt", model_name, re.IGNORECASE)
            or re.search("bloom", model_name, re.IGNORECASE)
            or re.search("mpt", model_name, re.IGNORECASE)
            or re.search("opt", model_name, re.IGNORECASE)
        ):
            is_graph_optimized = True
        # Move inputs to target device(s)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        # TODO there is an issue when do_sample is set to True for Habana
        generation_config.do_sample = False
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = is_graph_optimized

        def generate_output():
            with torch.no_grad():
                return model.generate(
                    **input_tokens,
                    **generate_kwargs,
                    streamer=streamer,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    lazy_mode=True,
                    hpu_graphs=use_hpu_graphs,
                )

        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    else:
        raise ValueError(
            f"Unsupported device type {device}, only supports cpu and hpu now."
        )
    for new_text in streamer:
        if len(new_text) == 0:
            continue
        yield new_text


def predict(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "mosaicml/mpt-7b-chat".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 1
    model_name = (
        params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_kv_cache"] if "use_kv_cache" in params else False

    prompt = params["prompt"]
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]

    if device == "cpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        stop_token_ids = [model.generation_config.eos_token_id]
        stop_token_ids.append(tokenizer(".", return_tensors="pt").input_ids)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )

        with torch.no_grad():
            with torch.cpu.amp.autocast(
                enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):
                generation_kwargs = dict(
                    generation_config=generation_config, return_dict_in_generate=True
                )
                generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [
                        StopOnTokens(
                            min_length=max(max_new_tokens - 20, 0),
                            start_length=input_token_len,
                            stop_token_id=stop_token_ids,
                        )
                    ]
                )
                generation_output = model.generate(**input_tokens, **generation_kwargs)
    elif device == "hpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_input_len(model, max_new_tokens),
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        stop_token_ids = [model.generation_config.eos_token_id]
        stop_token_ids.append(tokenizer(".", return_tensors="pt").input_ids)
        generate_kwargs = {
            "stopping_criteria": StoppingCriteriaList(
                [
                    StopOnTokens(
                        min_length=max(max_new_tokens - 20, 0),
                        start_length=input_token_len,
                        stop_token_id=stop_token_ids,
                    )
                ]
            )
        }
        is_graph_optimized = False
        if (
            re.search("gpt", model_name, re.IGNORECASE)
            or re.search("bloom", model_name, re.IGNORECASE)
            or re.search("mpt", model_name, re.IGNORECASE)
            or re.search("opt", model_name, re.IGNORECASE)
        ):
            is_graph_optimized = True
        # Move inputs to target device(s)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        # TODO there is an issue when do_sample is set to True for Habana
        generation_config.do_sample = False
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = is_graph_optimized

        with torch.no_grad():
            generation_output = model.generate(
                **input_tokens,
                **generate_kwargs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                lazy_mode=True,
                hpu_graphs=use_hpu_graphs,
            )
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    if "### Response:" in output:
        return output.split("### Response:")[1].strip()
    return output


def main():
    args = parse_args()
    base_model_path = args.base_model_path
    peft_model_path = args.peft_model_path
    prompts = create_prompts(
        [{"instruction": instruction, "input": ""} for instruction in args.instructions]
    )

    # Check the validity of the arguments
    if not 0 < args.temperature <= 1.0:
        raise ValueError("Temperature must be between 0 and 1.")
    if not 0 <= args.top_p <= 1.0:
        raise ValueError("Top-p must be between 0 and 1.")
    if not 0 <= args.top_k <= 200:
        raise ValueError("Top-k must be between 0 and 200.")
    if not 1.0 <= args.repetition_penalty <= 2.0:
        raise ValueError("Repetition penalty must be between 1 and 2.")
    if not 0 <= args.num_beams <= 8:
        raise ValueError("Number of beams must be between 0 and 8.")
    if not 32 <= args.max_new_tokens <= 1024:
        raise ValueError(
            "The maximum number of new tokens must be between 32 and 1024."
        )

    # User can use DeepSpeed to speedup the inference On Habana Gaudi processors.
    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    # For multi node, the value of the env variable WORLD_SIZE should be larger than 8
    use_deepspeed = (
        "deepspeed" in os.environ["_"]
        or ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 8)
        and args.habana
    )

    if args.habana:
        if use_deepspeed:
            # Set necessary env variables
            os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
            os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

        # Device is HPU
        args.device = "hpu"
        import habana_frameworks.torch.hpu as torch_hpu

        # Get world size, rank and local rank
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, args.local_rank = initialize_distributed_hpu()

        if use_deepspeed:
            # Check if DeepSpeed is installed
            from transformers.deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError(
                    "This script requires deepspeed: `pip install"
                    " git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0`."
                )
            import deepspeed

            # Initialize process(es) for DeepSpeed
            deepspeed.init_distributed(dist_backend="hccl")
            logger.info("DeepSpeed is enabled.")
        else:
            logger.info("Single-device run.")

        # Tweak generation so that it runs faster on Gaudi
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        adapt_transformers_to_gaudi()
        # Set seed before initializing model.
        from optimum.habana.utils import set_seed

        set_seed(args.seed)

    tokenizer_path = (
        args.tokenizer_name if args.tokenizer_name is not None else base_model_path
    )

    if use_deepspeed:
        with deepspeed.OnDevice(dtype=torch.bfloat16, device="cpu"):
            load_model(
                base_model_path,
                tokenizer_path,
                device="hpu",
                use_hpu_graphs=args.use_hpu_graphs,
                cpu_jit=args.jit,
                use_cache=args.use_kv_cache,
            )
            model = MODELS[base_model_path]["model"]
            if peft_model_path:
                model = PeftModel.from_pretrained(model, peft_model_path)
            model = model.eval()
            # Initialize the model
            ds_inference_kwargs = {"dtype": torch.bfloat16}
            ds_inference_kwargs["tensor_parallel"] = {"tp_size": 8}
            ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs
            # Make sure all devices/nodes have access to the model checkpoints
            torch.distributed.barrier()
            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
            model = deepspeed.init_inference(model, **ds_inference_kwargs)
            model = model.module
            MODELS[base_model_path]["model"] = model
    else:
        load_model(
            base_model_path,
            tokenizer_path,
            device="hpu" if args.habana else "cpu",
            use_hpu_graphs=args.use_hpu_graphs,
            cpu_jit=args.jit,
            use_cache=args.use_kv_cache,
        )

    if args.habana and rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(f"device: {args.device}, n_hpu: {world_size}, bf16")

    # warmup, the first time inference take longer because of graph compilation
    start_time = time.time()
    print("Warmup, Response: ")
    for new_text in predict_stream(
        model_name=base_model_path,
        device="hpu" if args.habana else "cpu",
        prompt="Tell me about Intel Xeon.",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0.0,
        use_hpu_graphs=args.use_hpu_graphs,
        use_cache=args.use_kv_cache,
        num_return_sequences=args.num_return_sequences,
    ):
        if args.local_rank in [-1, 0]:
            print(new_text, end="", flush=True)
    logger.info(f"duration: {time.time() - start_time}")

    for idx, tp in enumerate(zip(prompts, args.instructions)):
        prompt, instruction = tp
        idxs = f"{idx+1}"
        logger.info("=" * 30 + idxs + "=" * 30)
        logger.info(f"Instruction: {instruction}")
        start_time = time.time()
        logger.info("Response: ")
        for new_text in predict_stream(
            model_name=base_model_path,
            device="hpu" if args.habana else "cpu",
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            use_hpu_graphs=args.use_hpu_graphs,
            use_cache=args.use_kv_cache,
            num_return_sequences=args.num_return_sequences,
        ):
            if args.local_rank in [-1, 0]:
                print(new_text, end="", flush=True)
        logger.info(f"duration: {time.time() - start_time}")
        logger.info("=" * (60 + len(idxs)))

    for idx, tp in enumerate(zip(prompts, args.instructions)):
        prompt, instruction = tp
        idxs = f"{idx+1}"
        logger.info("=" * 30 + idxs + "=" * 30)
        logger.info(f"Instruction: {instruction}")
        start_time = time.time()
        logger.info("Response: ")
        out = predict(
            model_name=base_model_path,
            device="hpu" if args.habana else "cpu",
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            use_hpu_graphs=args.use_hpu_graphs,
            use_cache=args.use_kv_cache,
            num_return_sequences=args.num_return_sequences,
        )
        if args.local_rank in [-1, 0]:
            print(f"nonstream out = {out}")
        logger.info(f"duration: {time.time() - start_time}")
        logger.info("=" * (60 + len(idxs)))


if __name__ == "__main__":
    main()
