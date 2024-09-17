from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
print(amnesty_qa)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from ragas import evaluate

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

print(result)

df = result.to_pandas()
df.head()