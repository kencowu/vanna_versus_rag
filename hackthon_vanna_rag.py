import vanna_calls
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_response(df, user_prompt, api_key=api_key, model="gpt-3.5-turbo"):
    """
    Make a GPT API call with both system (df) and user prompts.

    Parameters:
    system_prompt (str): The system prompt to set the context for the conversation.
    user_prompt (str): The user prompt to ask the specific question or input.
    api_key (str): Your OpenAI API key.
    model (str): The model to use for the API call. Default is "gpt-4".

    Returns:
    str: The response from the GPT model.
    """

    system_prompt = f"You are a helpful data assistant. \n\nUser will ask you questions about the data according to the following  pandas DataFrame: \n{df.to_markdown()}\n\n"

    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model=model,
        # max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}         
        ]
    )
    
    # Extract the text from the response
    return response.choices[0].message["content"]


    # message_log = [
    #     self.system_message(
    #         f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
    #     ),
    #     self.user_message(
    #         "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
    #         self._response_language()
    #     ),
    # ]

    # summary = self.submit_prompt(message_log, **kwargs)

    # return summary