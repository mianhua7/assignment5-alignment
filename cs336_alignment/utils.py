from os.path import splitdrive
from typing import Any, Literal
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd

SYSTEM_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.
# Query:
```{instruction}```
# Answer:
```"""

MMLU_PROMPT = """Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).
Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:"""


def build_system_prompt(instruction: str) -> str:
    return SYSTEM_PROMPT.format(instruction=instruction)

def build_mmlu_prompt(subject: str, question: str, options: list[str]) -> str:
    assert len(options) == 4
    return MMLU_PROMPT.format(
        subject=subject,
        question=question,
        options=options,
    )

# format mmlu examples to mmlu_prompt_template
def format_mmlu_example(example: dict[str, Any]) -> dict[str, Any]:
    """Format MMLU examples to MMLU prompt template. Returns a dict for ds.map()."""
    instruction = build_mmlu_prompt(example["subject"], example["question"], example["options"])
    prompt = build_system_prompt(instruction)
    return {"prompt": prompt, "answer": example["answer"]}

def parse_mmlu_response(model_output: str) -> str | None:
    """Parse the model output into a predicted option letter (i.e., 'A', 'B', 'C', or 'D')."""
    try:
        answer = model_output.split("The correct answer is ")[-1].split(".")[0].strip()
        if answer in ["A", "B", "C", "D"]:
            return answer
        return None
    except (IndexError, AttributeError):
        return None

# load mmlu dataset from ../data/mmlu/dev/*csv files
def load_mmlu_dataset(
    split: Literal["dev", "test", "val"],
) -> Dataset:
    """Load MMLU dataset from ../data/mmlu/dev/*csv files."""
    records = []
    parent_dir = Path(__file__).resolve().parent.parent
    data_dir = parent_dir / "data" / "mmlu" / split
    paths = sorted(data_dir.glob("*.csv"))
    for p in paths:
        subject = p.stem.replace(f"_{split}", "")
        df = pd.read_csv(
            p,
            header=None,
            names=["question", "A", "B", "C", "D", "answer"],
        )
        for row in df.itertuples(index=False):
            records.append({
                "subject": subject,
                "question": row.question,
                "options": [row.A, row.B, row.C, row.D],
                "answer": row.answer,
            })
    return Dataset.from_list(records)
