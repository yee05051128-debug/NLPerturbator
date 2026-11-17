from .api_client import chat


def convert_to_question(text):
    prompt = f"""You task is to change the imperative sentence in the docstring to an interrogative sentence. \
You should only change one sentence in the docstring and keep the rest such as restrictions and conditions unchanged. \
Do not change the format such as line breaks ans whitespaces.

For example:
[original docstring]
For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |

[modified prompt]
For a given list of input numbers, can you calculate Mean Absolute Deviation
    around the mean of this dataset?
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |

Now it's your turn:
[original docstring]
{text}

[modified prompt]
"""
    return chat(prompt, temperature=0.2, max_tokens=256).rstrip(
        "[Modified prompt]\n"
    )


def perturbate(prompt: str, seed):
    """
    P2 - declarative_to_interrogative

    Algorithm: LLM
    """
    return convert_to_question(prompt)
