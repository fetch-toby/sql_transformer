import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import secret_key  # Assuming the secret_key.py file contains the access token

# Fetch the Hugging Face access token from secret_key.py
hf_token = secret_key.hugging_face_access_token

if hf_token is None:
    raise ValueError("Hugging Face access token not set in secret_key.py.")

# Load the model and tokenizer fine-tuned for SQL generation
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=hf_token, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=hf_token)

def generate_sql(query):
    # Preprocess the input query to make it suitable for the model
    input_text = f"translate English to SQL: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate SQL query using the model with a limit on new tokens
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100)

    # Decode the generated SQL query into a readable string
    sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql_query

# Example usage of the function
if __name__ == "__main__":
    input_query = "What is the name of the employee who works in the sales department?"
    sql_result = generate_sql(input_query)
    print(f"Generated SQL query: {sql_result}")
