import re
from dotenv import load_dotenv
import os
from genai import Client
from genai.credentials import Credentials
from genai import Client, Credentials
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
import json



def stock_entity_extraction_prompt(query):
    prompt_header = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are an entity extraction system that extracts stock-related information from a given query.
Here's a query:
"""
    prompt_footer = """
Extract the stock names and their corresponding stock symbols (if mentioned) in JSON format.
If a stock name is mentioned without its symbol, or vice versa, include the information that is provided.
If multiple stocks are mentioned, include all of them.
If no stocks are mentioned, return an empty list.

The format should be as follows:
{
    "stocks": [
        {
            "name": "Apple",
            "symbol": "AAPL"
        },
        {
            "name": "Google",
            "symbol": GOOG
        },
        {
            "name": null,
            "symbol": "MSFT"
        }
    ]
}

DO NOT RETURN ANYTHING OTHER THAN THE OUTPUT JSON
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_header + query + '\n' + prompt_footer


def json_extractor(input_str):
    # Regular expression to match from the first '{' to the last '}'
    pattern = r"\{.*\}"

    # Using re.DOTALL to make '.' match newlines as well
    match = re.search(pattern, input_str, re.DOTALL)

    if match:
        json_str = match.group(0)
        return json_str
    else:
        # raise ValueError("No valid JSON string found")
        return None


def get_generated_text(model_name, txt):
    # Load environment variables from .env file
    load_dotenv()

    # Accessing variables
    bam_api_key = os.getenv('GENAI_KEY')
    bam_url = os.getenv('BAM_URL')

    credentials = Credentials(
        api_key = bam_api_key,
        api_endpoint = bam_url
        )
    client = Client(credentials=credentials)
    # or if you want to pass values directly
    # credentials = Credentials(api_key="MY_API_KEY", api_endpoint="MY_ENDPOINT")

    prompt_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    question_end = "<|eot_id|>"
    prompt_footer = "<|start_header_id|>assistant<|end_header_id|>"

    txt = prompt_header + '\n' + txt + question_end + '\n' + prompt_footer

    response = list(
        client.text.generation.create(
            model_id=model_name,
            inputs=[txt],
            parameters=TextGenerationParameters(
                temperature=0,
                max_new_tokens=100,
                return_options=TextGenerationReturnOptions(input_text=True),
            ),
        )
    )
    result = response[0].results[0]
    return result.generated_text

def entity_extraction(query):
    prompt = stock_entity_extraction_prompt(query)
    response = get_generated_text(
        "meta-llama/llama-3-70b-instruct", prompt
    )
    # print("Response:", response)
    extracted_json = json_extractor(response)
    if extracted_json is None:
        return {
                "stocks":[]
                }
    return json.loads(extracted_json)

# query = 'Give me some fundamentals of the following stocks Google, Nvidia, Meta, Netflix?'
# entities  = entity_extrcation(query)