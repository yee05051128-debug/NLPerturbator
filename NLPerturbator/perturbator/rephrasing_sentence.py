import random
from .my_util import *
from .api_client import chat


def judge(token):
    end_symbols = [
        ".",
        "!",
        "?",
        ";",
        ":",
    ]
    return token.text in end_symbols


def paraphrase(sentence: str):
    prompt = f"""Rephrase the following programming-task sentence while preserving its exact meaning.
Keep identifiers, numbers, constraints, punctuation, and line breaks unchanged whenever possible.
Return only the rephrased sentence, without quotes or explanation.

Sentence:
{sentence}"""
    return chat(prompt, temperature=0.2, max_tokens=256)


def perturbate(prompt, times, seed):
    """
    P1 - rephrasing_sentence

    Algorithm: DL model
    """
    token_list = pro_tokenize(prompt)
    perturbed_text = ""


    perturbed_idx = [
        i
        for i, token in enumerate(token_list)
        if judge(token) or i == len(token_list) - 1
    ]
    perturbed_idx = random.sample(perturbed_idx, min(times, len(perturbed_idx)))

    last = 0
    for i, token in enumerate(token_list):
        if judge(token) or i == len(token_list) - 1:
            sentence = "".join([token.text for token in token_list[last : i + 1]])
            if i in perturbed_idx:
                sentence = paraphrase(sentence)
            last = i + 1
            perturbed_text += sentence
    return perturbed_text
