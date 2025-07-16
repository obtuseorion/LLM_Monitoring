import re
from typing import Any
from inspect_ai.dataset import Sample


def extract_answer(model_output):
    answer_marker = "ANSWER:"
    marker_position = model_output.find(answer_marker)
    
    if marker_position != -1:
        answer_start = marker_position + len(answer_marker)
        return model_output[answer_start:].strip()
    else:
        return ""
    
def find_code(completion: str) -> str:
    """Remove Markdown formatting around generated code blocks."""
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion

    return str(extracted_answer)


def record_to_sample(record: dict[str, Any]) -> Sample:
    # handle both field name possibilities
    prompt_field = "prompt" if "prompt" in record else "text"
    id_field = "id" if "id" in record else "task_id"
    
    return Sample(
        input=record[prompt_field],
        target=record["test_list"],
        id=record[id_field],
        metadata={
            "prompt": record[prompt_field],
            "test_list": record["test_list"],
            "test_list_str": "\n".join(record["test_list"]),
            "id": record[id_field],
            "backdoor": record["backdoor"]
        },
    )


"""
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["test_list"],
        id=record["id"],
        metadata={
            "prompt": record["prompt"],
            "test_list": record["test_list"],
            "test_list_str": "\n".join(record["test_list"]),
            "id": record["id"],
            "backdoor" : record["backdoor"]
        },
    )
"""

