"""The run_nars_lora notebook on a local GPU server instead of Colab: the same three systems, the same vLLM model,
adapter and dtype, the same results folder (running again resumes, as in the notebook).

    python common/run_lora_local.py MuSiQue --k 16              # one 16 GB GPU
    python common/run_lora_local.py MuSiQue --k 0 4 8 16 --gpus 2

Llama-3.1-8B in bfloat16 is 16 GB of weights, more than one 16 GB card holds, so with one GPU vLLM keeps part of the
weights in CPU memory (--cpu-offload-gb) and streams them in at every step: slower, but the numbers stay comparable
with the Colab runs (same bf16 weights; quantizing to FP8 would change the model). With two GPUs the model is split
across both (tensor parallel) and nothing is offloaded.
Each session appends its settings to <results>/runs.jsonl, so every answer can be traced to the machine that wrote it.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/common")
import experiment as ex

LORA_NAME = "NARS + 8B + LoRA v2 final"
ADAPTER_REVISION = "86e4c6d5f9abb2e3b06615bb8d0f01d2b6c8e76a"   # v2, as in the notebooks
MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
GPU_SETTINGS = {1: ["--cpu-offload-gb", "8", "--gpu-memory-utilization", "0.92"],
                2: ["--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.90"]}


def start_vllm(adapter, gpus, log):
    command = ["vllm", "serve", MODEL, "--served-model-name", "llama-3.1-8b-instruct", "--dtype", "bfloat16",
               "--enable-lora", "--lora-modules", f"nars-lora={adapter}", "--max-lora-rank", "16",
               "--max-model-len", "16384", "--port", "8000", *GPU_SETTINGS[gpus]]
    server = subprocess.Popen(command, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for minute in range(40):
        time.sleep(30)
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            print("vLLM is up")
            return server, command
        except Exception:
            if server.poll() is not None:
                print(open(log).read()[-3000:])
                raise SystemExit("vLLM stopped: see the log above")
            print(f"starting vLLM... {(minute + 1) / 2:.1f} min")
    raise SystemExit(f"vLLM did not start in 20 minutes: see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=list(ex.K_VALUES))
    parser.add_argument("--k", type=int, nargs="+", help="default: every k of the dataset")
    parser.add_argument("--n", type=int, help="questions (default: 500 for MuSiQue, 300 otherwise)")
    parser.add_argument("--gpus", type=int, default=1, choices=list(GPU_SETTINGS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ona", default=os.path.expanduser("~/neunars_test/ONA/NAR"), help="ONA built at dc4efd0")
    args = parser.parse_args()
    k_values = args.k or ex.K_VALUES[args.dataset]
    n = args.n or (500 if args.dataset == "MuSiQue" else 300)
    assert os.path.exists(args.ona), f"{args.ona} is missing: build ONA at dc4efd0 (sh build.sh)"

    from huggingface_hub import snapshot_download
    from openai import OpenAI
    os.chdir(f"{ROOT}/{args.dataset}")
    results = f"{ROOT}/{args.dataset}/results_final"
    adapter = snapshot_download(ex.ADAPTER_REPO, revision=ADAPTER_REVISION,
                                local_dir=os.path.expanduser(f"~/adapters/nars-lora-{ADAPTER_REVISION[:8]}"))
    server, command = start_vllm(adapter, args.gpus, f"{results}/vllm.log")
    try:
        questions, question_hash = ex.load_questions("questions.pkl", n)
        vllm = OpenAI(api_key="none", base_url="http://localhost:8000/v1", max_retries=2, timeout=1800)
        ex.check_models([(vllm, "nars-lora"), (vllm, "llama-3.1-8b-instruct"), (ex.deepinfra(), ex.JUDGE_MODEL)])
        systems = {"Llama-3.1-8B": (vllm, "llama-3.1-8b-instruct", None),
                   "Llama-3.1-8B + CoT": (vllm, "llama-3.1-8b-instruct", ex.COT),
                   LORA_NAME: (vllm, "nars-lora", args.ona)}
        ex.SIZES.setdefault(LORA_NAME, "8B")
        ex.NARS_SYSTEMS.add(LORA_NAME)
        with open(f"{results}/runs.jsonl", "a") as f:
            f.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(),
                                "k": k_values, "n": n, "adapter_revision": ADAPTER_REVISION, "vllm": command[2:]}) + "\n")
        compare = [ex.REFERENCE, f"{ex.REFERENCE} + CoT", "NARS + 8B + LoRA v2", LORA_NAME]
        for k in k_values:
            ex.run_all(questions, question_hash, results, systems, [k], workers=args.workers)
            print()
            ex.compare_k(results, compare, k, n)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
