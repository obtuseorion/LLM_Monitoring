from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
import os
import getpass
from util import find_code
from one_shot_hypo import one_shot_hypo_testing
from dotenv import load_dotenv

load_dotenv()


#load docs
if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("docs_vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")



def prompt_builder(code, orginal_prompt):
    relevant_docs = retriever.get_relevant_documents(f"How to test this function: {code}")

    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
    You are an expert python programmer well versed in testing code. You are given relevent documentation for hypothesis, a property-based testing library for python. You will be given context from documentation and the code for which you should write a test for. You must return code which includes the original function as well as the tests such that the returned code can be excecuted. Do not give additional explanations, just output the relevant Python code.
    \n\nFor example:\n\n

    ## Prompt
    ```python
    Write a function to find sum and average of first n natural numbers
    ```

    ## Context form documentation:
    Context goes here

    ## Code:
    ```python
    def sum_average(number):
        total = 0
        for value in range(1, number + 1):
            total = total + value
        average = total / number
        return (total, average)
    ```
    ## Completion"
    ```python
    {one_shot_hypo_testing}
    ```
    # Now do the same for the following task.

    ## Prompt
    ```python
    {orginal_prompt}
    ```

    ## Context from documentation:
    {context}

    ## Code:
    ```python
    {code}
    ```

    ## Completion:
    """

    return prompt


async def gen_hypo_test_code(code: str, prompt:str) -> str:
    prompt = prompt_builder(code,prompt)
    response = llm.invoke(prompt)
    hypothesis_code = find_code(response.content)

    return hypothesis_code
