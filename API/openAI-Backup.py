import openai

# Set your OpenAI API key
openai.api_key = "your-api-key-here" #Do not set it on the github repository

#mdoel can be changed
# potential models (check pricing and stuff) text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001", "gpt-4", "gpt-4-0314", "gpt-4-0613
def generate_code(prompt, model="gpt-4", max_tokens=300, temperature=0.2):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

prompt = "Physics prompt TBD"
generated_code = generate_code(prompt)
print(generated_code)
