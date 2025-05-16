# Scheming_LLM_Monitor

Preliminary work on a prompt reconstructor as a monitor for determining LM scheming


This script uses Inspect AI. Learn more about it here: https://inspect.aisi.org.uk/
```
pip install inspect-ai
```
The .env file assumes you are using Claude, if you prefer another LLM, either edit the .env file or just it ignore and set up with
```
export API_KEY=your-LLM-api-key
inspect eval arc.py --model LLM/model![SLM comparison](https://github.com/user-attachments/assets/93792f89-b20a-40da-a62d-c17a5536f2ae)

```
![SLM comparison](https://github.com/user-attachments/assets/ad3a5ce6-95d4-4e03-bff1-99b61507ec4c)

