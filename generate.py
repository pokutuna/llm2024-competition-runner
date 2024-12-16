import argparse
import re

import ollama
import polars as pl
from tqdm import tqdm

template_llm_jp = """

### 指示:
{{ range .Messages }}{{ .Content }}{{ end }}

### 応答:
"""


def run(
    df_tasks: pl.DataFrame,
    model: str,
    sub_model: str,
    outfile: str,
    ollama_host: str | None,
) -> None:
    print("Start generation")
    client = ollama.Client(host=ollama_host)

    tasks = df_tasks.to_dicts()
    outputs = []

    # まず一通り生成
    print(f"step1. Generate outputs with {model=}")
    for row in tqdm(tasks):
        output = generate(client, model, row["input"])
        outputs.append(output)
    write_outputs(df_tasks, outputs, outfile)

    # ことわざ・慣用句に関する問題はサブの力を借りる
    print("step2. Improve outputs for Japanese tasks")
    generated_examples: dict[int, list[str]] = {}

    print(f"step2-1. Generate examples with {sub_model=}")
    # モデルの読み込み回数を減らすため先に全体舐めて例を作る
    for i, row in enumerate(tqdm(tasks)):
        if re.search(r"(?:ことわざ|諺|慣用句)", row["input"]):
            es = [generate(client, sub_model, row["input"]) for _ in range(3)]
            generated_examples[i] = es

    print(f"step2-2. Generate outputs with examples {model=}")
    # 例を使ってメインモデルで再生成
    for i, row in enumerate(tqdm(tasks)):
        if i not in generated_examples:
            continue
        es = generated_examples[i]
        examples = ["- " + re.sub(r"\n+", " ", e).strip() for e in es]
        prompt = f"{row['input'].strip()}\n\n### 参考出力\n" + "\n".join(examples)
        outputs[i] = generate(client, model, prompt)
    write_outputs(df_tasks, outputs, outfile)


def generate(client: ollama.Client, model: str, input: str) -> str:
    output = client.generate(
        model=model,
        prompt=input,
        template=template_llm_jp if "llm-jp" in model else "",
    )
    return output.response


def read_as_df(path: str) -> pl.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        return pl.read_ndjson(path)
    elif path.endswith(".parquet"):
        return pl.read_parquet(path)
    elif path.endswith(".csv"):
        return pl.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_outputs(df_tasks: pl.DataFrame, outputs: list[str], path: str) -> None:
    df_tasks.with_columns(output=pl.Series(outputs)).write_ndjson(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True, help="タスクファイル")
    parser.add_argument("--outfile", type=str, default="./output.jsonl", help="出力先")
    parser.add_argument(
        "--ollama-host",
        type=str,
        help="ollama サーバのホスト, 省略時はデフォルト",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="主に利用するモデル",
        default="hf.co/pokutuna/llm2024-competition:gemma2-9b-v11-Q6_K.gguf",
    )
    parser.add_argument(
        "--sub-model",
        type=str,
        help="サブで利用するモデル、model に比べことわざ・慣用句に強いもの",
        default="hf.co/pokutuna/llm2024-competition:llm-jp-3-13b-v2-Q6_K.gguf",
    )
    args = parser.parse_args()

    run(
        df_tasks=read_as_df(args.tasks),
        model=args.model,
        sub_model=args.sub_model,
        outfile=args.outfile,
        ollama_host=args.ollama_host,
    )
