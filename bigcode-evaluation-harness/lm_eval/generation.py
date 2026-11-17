import json
import re
from math import ceil

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_eval.utils import TokenizedDataset, complete_code


def extract_python_code(text: str) -> str:
    """Extract Python source from a chat-model response.

    API models do not always follow the requested output format.  Accept a
    complete fenced block, an unterminated fenced block, or plain source code.
    """
    fenced = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()

    opening_fence = re.search(
        r"```(?:python|py)?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if opening_fence:
        return text[opening_fence.end() :].strip()

    return text.strip()


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
        

def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]
    
    #####################
    ###### OPEN AI ######
    #####################
    if args.api == True:
        import asyncio
        import os
        from dotenv import find_dotenv, load_dotenv
        from openai import AsyncOpenAI

        load_dotenv(find_dotenv(usecwd=True))

        def first_env(*names):
            for name in names:
                value = os.getenv(name)
                if value:
                    return value.strip()
            return None

        api_key = first_env(
            "CLOSEAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"
        )
        base_url = first_env(
            "CLOSEAI_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"
        )
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set CLOSEAI_API_KEY, DEEPSEEK_API_KEY, "
                "or OPENAI_API_KEY."
            )
        if not base_url:
            raise RuntimeError(
                "Missing API base URL. Set CLOSEAI_BASE_URL, DEEPSEEK_BASE_URL, "
                "or OPENAI_BASE_URL."
            )
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=args.api_timeout,
            max_retries=3,
        )
        
        selected_docs = [
            dataset[i]
            for i in range(args.limit_start, args.limit_start + n_tasks)
        ]
        if args.tasks == "humaneval":        
            prompts = [
                "Return only executable Python source code that solves the "
                "following problem. Do not include Markdown fences, analysis, "
                "or explanations. Preserve the required function name and "
                "signature.\n\n"
                + task.get_prompt(doc).strip()
                for doc in selected_docs
            ]
        elif args.tasks == "mbpp":
            prompts = [
                "Return only executable Python source code that solves the "
                "following problem. Do not include Markdown fences, analysis, "
                "or explanations.\n\n"
                + task.get_prompt(doc).strip()
                for doc in selected_docs
            ]
        else:
            raise ValueError("API mode currently supports humaneval and mbpp only")

        async def generate_all():
            semaphore = asyncio.Semaphore(max(1, args.api_max_concurrency))

            async def generate_one(prompt):
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=args.max_length_generation,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    content = response.choices[0].message.content
                    if not content:
                        raise RuntimeError("The API returned an empty response.")
                    return extract_python_code(content)

            generations = []
            for task_index, prompt in enumerate(prompts):
                texts = await asyncio.gather(
                    *(generate_one(prompt) for _ in range(args.n_samples))
                )
                generations.append(
                    [
                        task.postprocess_generation(
                            text,
                            task_index + args.limit_start,
                            args.api,
                        )
                        for text in texts
                    ]
                )
                print(f"API progress: {task_index + 1}/{len(prompts)}")
            return generations

        return asyncio.run(generate_all())
    #####################
    ###### OPEN AI ######
    #####################

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    stopping_criteria = []
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    if task.stop_words and tokenizer.eos_token:
        task.stop_words.append(tokenizer.eos_token)    
    if hasattr(task, "check_fn"):
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
        )
    elif task.stop_words:
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer)
        )
    if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
        stopping_criteria.append(
            TooLongFunctionCriteria(0, task.max_length_multiplier)
        )
    
    if stopping_criteria:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        limit_start=args.limit_start,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
        instruction_tokens=instruction_tokens,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    if args.max_memory_per_gpu is not None:
        # The model is already sharded across multiple GPUs
        ds_loader = accelerator.prepare(ds_loader)
    elif not is_loaded_in_8bit and not is_loaded_in_4bit:
        # we only wrap data loader to avoid extra memory occupation
        model = model.to(accelerator.device)
        ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        limit_start=args.limit_start,
        batch_size=args.batch_size,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        is_wrapped=is_loaded_in_8bit or is_loaded_in_4bit,
        **gen_kwargs,
    )
    return generations
