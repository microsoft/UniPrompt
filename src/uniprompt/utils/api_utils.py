import json
import os
import re
import sqlite3
import time
from typing import Optional

from openai import AzureOpenAI, OpenAI


def extract_answer(cot: str) -> Optional[str]:
    pattern = r"<Answer>(.*?)<\/Answer>"
    match = re.search(pattern, cot, re.DOTALL)
    answer = match.group(1) if match else cot
    answer = answer.strip(";")
    return answer

def chat_completion(cache_path=None, **kwargs):
    def make_api_call(client, **kwargs):
        while True:
            try:
                response = client.chat.completions.create(**kwargs["model_kwargs"], messages=kwargs["messages"])
                return response.choices[0].message.content
            except Exception as e:
                print(f"Got error {e}. Sleeping for 5 seconds...")
                time.sleep(5)

    if kwargs["api_kwargs"]["api_key"] is not None:
        api_key = kwargs["api_kwargs"]["api_key"]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if kwargs["api_kwargs"]["api_type"] == "azure":
        client = AzureOpenAI(api_key=api_key, azure_endpoint=kwargs["api_kwargs"]["api_base"], api_version=kwargs["api_kwargs"]["api_version"])
    elif kwargs["api_kwargs"]["api_base"]:
        client = OpenAI(base_url=kwargs["api_kwargs"]["api_base"], api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)

    if not cache_path:
        return make_api_call(client, **kwargs)

    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    # Connect to SQLite database
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_cache
    (model TEXT, messages TEXT, response TEXT)
    """)

    # Create a unique key from model and messages
    model = json.dumps(kwargs["model_kwargs"])
    messages = json.dumps(kwargs["messages"])

    # Check if the response is in the cache
    cursor.execute("SELECT response FROM api_cache WHERE model = ? AND messages = ?", (model, messages))
    cached_response = cursor.fetchone()

    if cached_response:
        conn.close()
        return cached_response[0]

    # If not in cache, make the API call
    response_content = make_api_call(client, **kwargs)

    # Store the response in the cache
    cursor.execute("INSERT INTO api_cache (model, messages, response) VALUES (?, ?, ?)",
                   (model, messages, response_content))
    conn.commit()
    conn.close()

    return response_content