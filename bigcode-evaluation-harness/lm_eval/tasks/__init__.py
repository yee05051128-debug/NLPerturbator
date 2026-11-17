import inspect
import os
from pprint import pprint

from . import (
    apps,
    codexglue_code_to_text,
    codexglue_text_to_text,
    conala,
    concode,
    gsm,
    humaneval,
    humanevalpack,
    instruct_humaneval,
    mbpp,
    parity,
    python_bugs,
    quixbugs,
    recode,
)

# DS-1000 depends on the Unix-only ``fcntl`` module.  Keep the rest of the
# task registry usable on Windows, which is sufficient for HumanEval/MBPP.
if os.name != "nt":
    from . import ds1000, multiple

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **humaneval.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
}

if os.name != "nt":
    TASK_REGISTRY.update(ds1000.create_all_tasks())
    TASK_REGISTRY.update(multiple.create_all_tasks())

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
