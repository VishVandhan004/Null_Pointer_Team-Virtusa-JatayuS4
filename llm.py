from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

def get_loan_explanation(predicted_rate, input_data):
    user_prompt = f"""
The predicted home loan interest rate is exactly {predicted_rate:.2f}% based on the customer profile below:

{input_data}

Return a clear, concise explanation of ONLY the key factors that specifically contributed to this {predicted_rate:.2f}% rate.

- Use one bullet per relevant factor only.
- Skip any factor that had minimal or no effect.
- Each explanation must be a **short, single sentence**, and **directly related to the data**.
- DO NOT include general financial advice, summaries, disclaimers, or change the rate value.

Output format (only include actual factors that impacted the prediction):

- [Factor Name]: [Short one-line impact reason]
"""

    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[
            {"role": "system", "content": "You are a financial assistant specialized in explaining home loan interest rates in precise language."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        top_p=0.9,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

def get_fd_explanation(predicted_rate_adjustment, input_data):
    user_prompt = f"""
The predicted Fixed Deposit interest rate adjustment is exactly {predicted_rate_adjustment:.2f} bps based on the customer profile below:

{input_data}

Return a clear, concise explanation of ONLY the key factors that specifically contributed to this {predicted_rate_adjustment:.2f} bps adjustment.

- Use one bullet per relevant factor only.
- Skip any factor that had minimal or no effect.
- Each explanation must be a **short, single sentence**, and **directly related to the data**.
- DO NOT include general financial advice, summaries, disclaimers, or change the rate adjustment value.

Output format (only include actual factors that impacted the prediction):

- [Factor Name]: [Short one-line impact reason]
"""

    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[
            {"role": "system", "content": "You are a financial assistant specialized in explaining Fixed Deposit interest rate adjustments in precise language."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        top_p=0.9,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()