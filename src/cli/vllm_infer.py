# Copyright 2025 the LlamaFactory team.
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

import ast
import gc
import json
import os
from typing import Optional, Union



import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    enum_values: Optional[Union[str, list]] = None,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage:
        python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo

        With enum_values for structured output (multiple formats supported):

        # JSON format with double quotes (recommended)
        python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --enum_values '["Layer 1", "Layer 2/3", "Layer 5", "Layer 6"]'

        # Comma-separated (simplest)
        python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --enum_values "Layer 1,Layer 2/3,Layer 5,Layer 6"


    """
    import sys
    print("=" * 80, flush=True)
    print(f"Starting vLLM inference...", flush=True)
    print(f"Model: {model_name_or_path}", flush=True)
    print(f"Output file: {save_name}", flush=True)
    if enum_values:
        print(f"Using structured output with enum_values", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    # Parse enum_values from CLI input
    parsed_enum_values = None
    if enum_values is not None:
        # If it's already a list (fire sometimes converts it), use it directly
        if isinstance(enum_values, (list, tuple)):
            parsed_enum_values = list(enum_values)
        elif isinstance(enum_values, str):
            try:
                # Try to parse as JSON first (for list format)
                parsed_enum_values = json.loads(enum_values)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to parse as Python literal (for set/list format)
                try:
                    parsed_enum_values = ast.literal_eval(enum_values)
                    # Convert set to list if needed
                    if isinstance(parsed_enum_values, set):
                        parsed_enum_values = list(parsed_enum_values)
                except (ValueError, SyntaxError):
                    # If all parsing fails, split by comma as fallback
                    parsed_enum_values = [v.strip() for v in enum_values.split(',')]
        else:
            raise TypeError(f"enum_values must be a string, list, or tuple, got {type(enum_values)}")

    if parsed_enum_values is not None:
        response_format = {
            '$defs': {
                'MicroenvironmentType': {
                    'enum': parsed_enum_values,
                    'title': 'MicroenvironmentType',
                    'type': 'string'
                }
            },
            'properties': {'microenvironment': {'$ref': '#/$defs/MicroenvironmentType'}},
            'required': ['microenvironment'],
            'title': 'MicroenvironmentPrediction',
            'type': 'object'
        }
        structured_outputs_params_json = GuidedDecodingParams(json=response_format)
    else:
        structured_outputs_params_json = None


    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        guided_decoding=structured_outputs_params_json,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    #Initialize output file
    print(f"Output file: {save_name}", flush=True)
    with open(save_name, "w", encoding="utf-8") as f:
        pass

    print(f"Dataset size: {len(train_dataset)} samples", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Starting inference...", flush=True)

    total_samples = 0
    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        for j in range(len(batch["input_ids"])):
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
            else:
                multi_modal_data = None

            vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
            prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        # Save results incrementally
        with open(save_name, "a", encoding="utf-8") as f:
            for text, pred, label in zip(prompts, preds, labels):
                f.write(json.dumps({"prompt": "prompt is in dataset", "predict": pred, "label": label}, ensure_ascii=False) + "\n")

        total_samples += len(prompts)
        gc.collect()

    print("*" * 70, flush=True)
    print(f"✓ Successfully saved {total_samples} results to {save_name}", flush=True)
    print("*" * 70, flush=True)

    # Cleanup: properly destroy the LLM engine and free resources
    print("Cleaning up resources...", flush=True)

    # Set a watchdog timer to force exit if cleanup takes too long
    import threading
    import os
    import time

    def force_exit_timer():
        time.sleep(5)
        print("Cleanup timed out. Forcing exit...", flush=True)
        os._exit(0)

    daemon_thread = threading.Thread(target=force_exit_timer, daemon=True)
    daemon_thread.start()

    try:
        # First, try to explicitly shutdown the executor if it exists
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            executor = llm.llm_engine.model_executor
            if hasattr(executor, 'shutdown'):
                executor.shutdown()

        del llm
        gc.collect()

        # Note: We skip dist.destroy_process_group() as it can hang.
        # os._exit(0) is sufficient to cleanup resources.

        # Clean up any Ray resources if Ray is being used
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass

    except Exception as e:
        print(f"Warning during cleanup: {e}", flush=True)

    print("Inference complete. Exiting...", flush=True)
    os._exit(0)



if __name__ == "__main__":
    fire.Fire(vllm_infer)