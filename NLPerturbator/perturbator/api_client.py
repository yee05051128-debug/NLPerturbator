import os

from openai import OpenAI
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv(usecwd=True))


def _first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def get_api_key():
    api_key = _first_env(
        "CLOSEAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set CLOSEAI_API_KEY, DEEPSEEK_API_KEY, "
            "or OPENAI_API_KEY."
        )
    return api_key


def get_base_url():
    base_url = _first_env(
        "CLOSEAI_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "OPENAI_BASE_URL",
    )
    if not base_url:
        raise RuntimeError(
            "Missing API base URL. Set CLOSEAI_BASE_URL, DEEPSEEK_BASE_URL, "
            "or OPENAI_BASE_URL."
        )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def get_model():
    return _first_env("CLOSEAI_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL") or "deepseek-chat"


def chat(prompt, *, temperature=0.2, max_tokens=512):
    client = OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=120.0,
        max_retries=3,
    )
    response = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The API returned an empty response.")
    return content.strip()
