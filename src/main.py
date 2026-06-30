import argparse
import functools
from types import SimpleNamespace
import warnings

import torch
import transformers

try:
    import wandb
except ImportError:
    wandb = None

try:
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table
    from lm_eval.models.huggingface import HFLM
except ImportError:
    lm_eval, TaskManager, make_table, HFLM = None, None, None, None

try:
    import platinumbench
except ImportError:
    platinumbench = None

from common_utils import clear_device_cache, fix_seed
from data_utils import get_data, get_wikitext2
from evaluate_perplexity import compute_perplexity
from quant_hf import WushQuantConfig
from quantize_model import quantize_model

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def parse_args() -> argparse.Namespace:

    def str2bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def auto_or_int(value: str) -> int | str:
        if value == "auto":
            return value
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # Misc params
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed.",
    )
    parser.add_argument(
        "--log_wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The name or path to the model to quantize. Not needed with --load_path.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=False,
        help="whether to use torch.compile.",
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["fp16", "mxfp4", "nvfp4"],
        help="Quantization format. 'fp16' runs the unquantized baseline (no quantization or transform). "
             "Not needed with --load_path (the format is read from the checkpoint).",
    )
    parser.add_argument(
        "--gptq",
        type=str2bool,
        default=True,
        help="Run GPTQ quantization.",
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        default="wush",
        choices=["i", "h", "wush"],
        help="Transform type: i (identity), h (Hadamard), wush (WUSH).",
    )
    parser.add_argument(
        "--dampen_ratio",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--activation_quantile",
        type=float,
        default=None,
        help="nvfp4 only, opt-in: if set, calibrate the activation global scale via a full calibration pass to "
             "the q-quantile of the transformed-activation magnitude (use 1.0 for amax; q<1 clips and is "
             "typically harmful). Default (unset) keeps the inherited weight global scale and runs no extra "
             "measurement pass. No effect on mxfp4.",
    )
    parser.add_argument(
        "--cpu_offload_modules",
        type=str2bool,
        default=True,
        help="whether to offload modules to CPU.",
    )
    parser.add_argument(
        "--cpu_offload_activations",
        type=str2bool,
        default=True,
        help="whether to offload activations to CPU.",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        default="fineweb-edu",
        type=str,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length",
        default=2048,
        type=int,
        help="Length of calibration sequences.",
    )
    parser.add_argument(
        "--num_sequences",
        default=1024,
        type=int,
        help="Number of calibration sequences.",
    )
    # Save / load params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the quantized model (via save_pretrained). Saving is triggered by setting this.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to a quantized checkpoint to load (via from_pretrained) and evaluate, skipping quantization.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=5 * 2 ** 30,
        help="Maximum shard size in bytes."
    )
    # Eval params
    parser.add_argument(
        "--eval_perplexity",
        type=str2bool,
        default=True,
        help="whether to eval perplexity after quantization.",
    )
    parser.add_argument(
        "--eval_openllm",
        type=str2bool,
        default=False,
        help="whether to eval OpenLLM v1 after quantization.",
    )
    parser.add_argument(
        "--eval_platinumbench",
        type=str2bool,
        default=False,
        help="whether to eval platinumbench after quantization.",
    )
    # LM eval params
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        type=str,
        default=["mmlu_cot_llama","mmlu","humaneval", "humaneval_instruct", "gsm8k_llama", "hellaswag", "winogrande"],
        help="OpenLLMv1 tasks to evaluate after quantization."
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_batch_size_mmlu_gsm8k",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    parser.add_argument(
        "--disable_thinking",
        type=str2bool,
        default=True,
        help="Whether to disable thinking mode for Qwen3.",
    )
    # Parse arguments
    args: argparse.Namespace = parser.parse_args()
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    # Check eval dependencies
    if args.eval_openllm:
        assert lm_eval is not None, "lm_eval is not installed. Please install lm_eval `pip install lm_eval`."
    if args.eval_platinumbench:
        assert platinumbench is not None, "platinumbench is not installed. Please install platinum-benchmarks."
    # Load mode (a pre-quantized checkpoint) vs quantize mode: in load mode the model and format come from the
    # checkpoint itself, so --model_name_or_path / --format are not needed.
    if not args.load_path:
        assert args.model_name_or_path is not None, \
            "Specify --model_name_or_path (to quantize) or --load_path (to load a quantized checkpoint)."
        assert args.format is not None, "--format is required when quantizing (omit it only with --load_path)."
    if args.format == "fp16" and args.save_path:
        warnings.warn("--save_path is ignored when --format=fp16: the unquantized baseline is not exported.")
    return args


def main() -> None:
    args: argparse.Namespace = parse_args()
    print(args)
    # Fix seed
    fix_seed(args.seed)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)
    # Set device
    device: torch.device = torch.device(torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda")
    # Model: either load a pre-quantized checkpoint (the 'wush' quantizer rebuilds QuantLinear and skips
    # quantization) or load the base model to quantize. The pre-quantized path places straight on the device.
    if args.load_path:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.load_path,
            torch_dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
            device_map=device,
            low_cpu_mem_usage=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.load_path)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            torch_dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
            device_map=None if args.cpu_offload_modules else device,
            low_cpu_mem_usage=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    model.config.use_cache = False
    model.requires_grad_(False)

    # Sanity check
    if args.eval_openllm:
        assert hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None, "OpenLLM v1 works only with chat template."
        if args.disable_thinking:
            if model.config.model_type == "qwen3":
                tokenizer.apply_chat_template = functools.partial(
                    tokenizer.apply_chat_template,
                    enable_thinking=False
                )
            else:
                warnings.warn("--disable_thinking has no effect on non-Qwen3 models.")

    if not args.load_path and args.format != "fp16":
        # Prepare calibration data
        calibration_data: list[torch.Tensor] = get_data(
            dataset_name=args.dataset_name_or_path,
            tokenizer=tokenizer,
            max_sequence_length=args.sequence_length,
            num_calibration_samples=args.num_sequences,
            seed=args.seed,
        )
        # Quantize (swaps QuantLinear into the model in place)
        quantize_model(
            model=model,
            calibration_data=calibration_data,
            quantize_format_name=args.format,
            transform_type=args.transform_type,
            dampen_ratio=args.dampen_ratio,
            activation_quantile=args.activation_quantile,
            force_rtn=not args.gptq,
            device=device,
            cpu_offload_modules=args.cpu_offload_modules,
            cpu_offload_activations=args.cpu_offload_activations,
        )
        # Save via transformers interfaces. The config carries only quant_method:"wush"; the quantized
        # tensors (and hence group size / scale dtype / transform) are recovered on load from the checkpoint.
        if args.save_path:
            model.config.quantization_config = WushQuantConfig()
            model.save_pretrained(args.save_path, max_shard_size=args.max_shard_size)
            tokenizer.save_pretrained(args.save_path)

    # Compile
    if args.compile:
        model = torch.compile(model)

    # Evaluate
    if args.eval_perplexity:
        model.to(device=device)
        eval_data: list[torch.Tensor] = get_wikitext2(tokenizer, args.sequence_length)
        ppl: float = compute_perplexity(model=model, data=eval_data, batch_size=1)
        print(f"Wikitext-2 perplexity: {round(ppl, 2):.2f}")
        if args.log_wandb:
            wandb.log({"eval/wikitext2_ppl": ppl})

    if args.eval_platinumbench:
        model.to(device=device)
        if args.log_wandb:
            uniquename: str = (wandb.run.name + str(args.seed)).replace("/", "--")
            print(f"Platinumbench unique name: {uniquename}")
        else:
            uniquename: str = str(args.seed)
        platinumbench.run_benchmark(
            model_list={uniquename:(model, tokenizer)},
            output_file=None,
            parallelism=1,
            errors_dir=None,
            seed=args.seed,
            wandb=wandb if args.log_wandb else None,
            args=SimpleNamespace(),
        )

    if args.eval_openllm:
        model.to(device=device)
        task_manager: TaskManager = TaskManager()
        results = {}

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.lm_eval_batch_size,
            max_length=4096, # from open LLM openllm
        )

        # Winogrande (5-shot)
        if "winogrande" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="winogrande",
                num_fewshot=5,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            if args.log_wandb:
                wandb.log({"eval/openllm": task_results})

        clear_device_cache(garbage_collection=True)

        # Hellaswag (10-shot)
        if "hellaswag" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="hellaswag",
                num_fewshot=10,
                batch_size= args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            if args.log_wandb:
                wandb.log({"eval/openllm": task_results})

        del lm
        clear_device_cache(garbage_collection=True)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.lm_eval_batch_size_mmlu_gsm8k,
            max_length=4096, # from open LLM openllm
        )

        # MMLU CoT
        if "mmlu_cot_llama" in args.lm_eval_tasks or "mmlu" in args.lm_eval_tasks:
            if model.config.model_type == "qwen3":
                print("Evaluating MMLU without CoT for Qwen3...")
                task_results = lm_eval.simple_evaluate(
                    model=lm,
                    tasks="mmlu",
                    batch_size=args.lm_eval_batch_size_mmlu_gsm8k,
                    task_manager=task_manager,
                )["results"]
                results.update(task_results)
                print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
                if args.log_wandb:
                    wandb.log({"eval/openllm": task_results})
            else:
                task_results = lm_eval.simple_evaluate(
                    model=lm,
                    tasks="mmlu_cot_llama",
                    batch_size=args.lm_eval_batch_size_mmlu_gsm8k,
                    apply_chat_template=True,
                    fewshot_as_multiturn=True,
                    task_manager=task_manager,
                )["results"]
                results.update(task_results)
                print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
                if args.log_wandb:
                    wandb.log({"eval/openllm": task_results})

        clear_device_cache(garbage_collection=True)

        # GSM8K
        if "gsm8k_llama" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="gsm8k_llama",
                batch_size=args.lm_eval_batch_size_mmlu_gsm8k,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            if args.log_wandb:
                wandb.log({"eval/openllm": task_results})

        del lm
        clear_device_cache(garbage_collection=True)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.lm_eval_batch_size,
            max_length=4096, # from open LLM openllm
        )

        if "humaneval" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                    model=lm,
                    tasks="humaneval",
                    batch_size=args.lm_eval_batch_size,
                    task_manager=task_manager,
                    confirm_run_unsafe_code=True,
                )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            if args.log_wandb:
                wandb.log({"eval/openllm": task_results})

        clear_device_cache(garbage_collection=True)

        if "humaneval_instruct" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="humaneval_instruct",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                confirm_run_unsafe_code=True,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            if args.log_wandb:
                wandb.log({"eval/openllm": task_results})

        clear_device_cache(garbage_collection=True)

        # Print formatted table
        print("### Final results ###")
        print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))


if __name__ == "__main__":
    main()
