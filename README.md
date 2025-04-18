# Scheming_LM_monitor_MATS
Preliminary work on a prompt reconstructor as a monitor for determining LM scheming


This script uses Inspect AI. Learn more about it here: https://inspect.aisi.org.uk/
```
pip install inspect-ai
```
The .env file assumes you are using Claude, if you prefer another LLM, either edit the .env file or just it ignore and set up with
```
export API_KEY=your-LLM-api-key
inspect eval arc.py --model LLM/model
```

CREDIT: Much of the prompt template code, the code execution code in the verify function, and the find code function was taken from the MBPP eval, see here: https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/mbpp . The python_questions were also sourced from the MBPP dataset (at random). See here for more details: https://huggingface.co/datasets/google-research-datasets/mbpp/viewer/sanitized/test?row=18&views%5B%5D=sanitized_test
