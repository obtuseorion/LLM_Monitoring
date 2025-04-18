# Scheming_LM_monitor_MATS
Preliminary work on a prompt reconstructor as a monitor for determining LM scheming


This script uses Inspect AI. Learn more about it here: https://inspect.aisi.org.uk/
```
pip install inspect-ai
```
The .env file assumes you are using Claude, if you prefer another LLM, either edit the .env file or just ignore and setup with
```
export API_KEY=your-LLM-api-key
inspect eval arc.py --model LLM/model
```
