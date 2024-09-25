import os
import dotenv
from time import sleep

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Maximum number of tokens that the openai api allows me to request per minute
RATE_LIMIT = 250000
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


# To avoid rate limits, we use exponential backoff where we wait longer and longer
# between requests whenever we hit a rate limit. Explanation can be found here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# I'm using default parameters here, I don't know if something else might be
# better.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


# Define a function that adds a delay to a Completion API call
def delayed_completion_with_backoff(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return completion_with_backoff(**kwargs)


def completion_create_retry(model, messages, sleep_time=5, **kwargs):
    """A wrapper around openai.Completion.create that retries the request if it fails for any reason."""
    #print("messages: ", messages)
    """
    if 'llama' in kwargs['model'] or 'vicuna' in kwargs['model'] or 'alpaca' in kwargs['model']:
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']
        return kwargs['endpoint'](prompts, **kwargs)
    else:
    """
    while True:
    
            try:
                # for gpt4 I need to update the call
                openai_response = client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                return openai_response
                #return client.completions.create(*args, **kwargs)
                #return openai.Completion.create(*args, **kwargs)
            except Exception as e:
                print('OpenAI error:', e)
                print('MESSAGES: ', messages)
                sleep(sleep_time)
