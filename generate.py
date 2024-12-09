import argparse

import ollama
import polars as pl
from tqdm import tqdm


def run(df_tasks: pl.DataFrame, model: str, host: str | None) -> pl.DataFrame:
    client = ollama.Client(host=host)
    outputs = []
    for row in tqdm(df_tasks.to_dicts()):
        output = client.generate(model, row["input"])
        outputs.append(output.response)
    return df_tasks.with_columns(output=pl.Series(outputs))


def read_as_df(path: str) -> pl.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        return pl.read_ndjson(path)
    elif path.endswith(".parquet"):
        return pl.read_parquet(path)
    elif path.endswith(".csv"):
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="./output.jsonl")
    parser.add_argument("--ollama-host", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="hf.co/pokutuna/llm2024-gemma2:gemma2-9b-sft009-Q6_K.gguf",
    )
    args = parser.parse_args()

    df_tasks = read_as_df(args.tasks)
    df_output = run(df_tasks, args.model, args.ollama_host)
    df_output.write_ndjson(args.outfile)
