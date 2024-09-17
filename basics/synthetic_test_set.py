from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("data/documents")
documents = loader.load()

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")#"gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

df_testset = testset.to_pandas()

print(df_testset.head())

