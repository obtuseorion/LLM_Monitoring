# LLM code generation monitoring methods evals

A suite of original LLM monitoring methods, evaluated via inspect AI. 
Currently:

**Prompt Reconstructor**: Generates a prompt based on the generated code, and this reconstructed prompt is compared to the original prompt. In the case of scheming, the reconstructed prompt would differ from the original prompt.

**Property-based test generator (Hypothesis testing)**: Generates property-based unit tests via a RAG pipeline. If the generated unit tests fail, the LLM is determined to be scheming.

This script uses Inspect AI. Learn more about it here: https://inspect.aisi.org.uk/
```
pip install inspect-ai
```
The .env file assumes you are using Claude, if you prefer another LLM, either edit the .env file or just set up with
```
export API_KEY=your-LLM-api-key
inspect eval arc.py --model LLM/model![SLM comparison](https://github.com/user-attachments/assets/93792f89-b20a-40da-a62d-c17a5536f2ae)
```

Run ```pip install -r requirements.txt``` to download dependencies.

To run an eval, navigate to the desired method and run the following command:
```
inspect eval task.py
```

