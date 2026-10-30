# llm.py: DeepInfra chat calls for the data scripts - parallel, retried, with token and cost accounting.
#
#   llm = DeepInfra()                                    # reads DEEPINFRA_API_KEY
#   replies = llm.ask_all(prompts, max_tokens=32, desc="Answering")   # one reply (or None) per prompt, in order
#   print(llm.summary())
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from keys import deepinfra_key
from tqdm import tqdm

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
PRICES = (0.40, 0.40)  # USD per million tokens: (input, output)


class DeepInfra:
    def __init__(self, model=MODEL, workers=16, prices=PRICES):
        self.client = OpenAI(api_key=deepinfra_key(), base_url="https://api.deepinfra.com/v1/openai",
                             timeout=60, max_retries=3)
        self.model, self.workers, self.prices = model, workers, prices
        self.requests = self.failed = self.tokens_in = self.tokens_out = 0
        self._lock = threading.Lock()

    def check(self):
        """One tiny request before a big run: stop at once on a bad key or model name instead of failing every call."""
        try:
            self.client.chat.completions.create(model=self.model, max_tokens=1,
                                                messages=[{"role": "user", "content": "Hi"}])
        except Exception as error:
            raise SystemExit(f"DeepInfra check failed for {self.model}: {type(error).__name__}: {error}")

    def ask(self, prompt, max_tokens):
        """One prompt -> the reply text, or None if the call still failed after retries."""
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
        except Exception as error:
            with self._lock:
                self.failed += 1
            tqdm.write(f"Request failed: {type(error).__name__}: {error}")
            return None
        with self._lock:
            self.requests += 1
            self.tokens_in += response.usage.prompt_tokens
            self.tokens_out += response.usage.completion_tokens
        return (response.choices[0].message.content or "").strip()

    def ask_all(self, prompts, max_tokens, desc):
        """Many prompts at once (self.workers in parallel); replies come back in the same order."""
        with ThreadPoolExecutor(self.workers) as pool:
            return list(tqdm(pool.map(lambda p: self.ask(p, max_tokens), prompts), total=len(prompts), desc=desc))

    def cost(self):
        return (self.tokens_in * self.prices[0] + self.tokens_out * self.prices[1]) / 1_000_000

    def summary(self):
        return {"model": self.model, "requests": self.requests, "failed": self.failed,
                "input_tokens": self.tokens_in, "output_tokens": self.tokens_out, "cost_usd": round(self.cost(), 4)}
