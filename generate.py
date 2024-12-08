import argparse

import ollama
import polars as pl
from tqdm import tqdm


def run(df_tasks: pl.DataFrame, model: str, host: str | None) -> pl.DataFrame:
    client = ollama.Client(host=host)
    outputs = []
    for row in tqdm(df_tasks.to_dicts()):
        output = client.generate(model, row["input"])
        outputs.append(output)
    return df_tasks.with_columns(outputs=pl.Series(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="./output.jsonl")
    parser.add_argument("--ollama-host", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="hf.co/pokutuna/llm2024-gemma2:gemma2-9b-sft005-Q6_K.gguf",
    )
    args = parser.parse_args()

    df_tasks = pl.read_ndjson(args.tasks)
    df_output = run(df_tasks, args.model, args.ollama_host)
    df_output.write_json(args.outfile)
