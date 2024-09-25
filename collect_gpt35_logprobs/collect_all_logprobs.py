import sys
sys.path.append('..')
sys.path.append('../')

import argparse
import os

import dotenv
import numpy as np
import openai

from lllm.questions_loaders import SyntheticFacts, Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture


def find_number_of_rows_with_questions_the_model_can_answer(number_answerable_questions_required, dataset, model):
    return np.where(dataset[f"{model}_can_answer"].cumsum() == number_answerable_questions_required)[0][0] + 1


dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='questions1000')

args = parser.parse_args()

print(f'Testing on {args.dataset}')


dataset_map = {'synthetic_facts': SyntheticFacts,
               'questions1000': Questions1000,
               'wikidata': WikiData,
               'common': Commonsense2,
               'engtofre': TatoebaEngToFre,
               'fretoeng': TatoebaFreToEng,
               'sciq': Sciq,
               'math': MathematicalProblems,
               'anthropic_aware_ai': AnthropicAwarenessAI,
               'anthropic_aware_arch': AnthropicAwarenessArchitecture,
               'anthropic_aware_nnarch': AnthropicAwarenessNNArchitecture,
               }

dataset = dataset_map[args.dataset]()
print('Dataset loaded.')

n_rows = len(dataset)

dataset.generate_logprobs_no_checks(
    model_suspect="gpt-3.5-turbo-instruct",
    regenerate_if_done_before=False,
    max_questions_to_try=None,
    # max_questions_to_try=10,
    save_progress=True,
)

print("GENERATE GPT 3.5 Turbo Instruct COMPLETED CORRECTLY")
