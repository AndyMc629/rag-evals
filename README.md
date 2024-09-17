# rag-evals
Playing with rag evals.

# Basics
Data from https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/tree/main/raw_data/text_data

To spin up the Postgres database go to infra/db/dockerfiles and run 

$docker compose up

... The usual good stuff.

I've set the devcontainer.json file to use the host's network with the following (obviously needs changed for real app's)

```
"runArgs": ["--network=host"], 
```
