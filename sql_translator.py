import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer for SQL generation
model_name = "t5-small"  # You can also use "t5-base" for larger models
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_sql(query):
    """
    Function to generate an SQL query from a natural language input using a T5 model.

    Args:
    query (str): The input question in natural language.

    Returns:
    str: The generated SQL query.
    """
    # Preprocess the input query to make it suitable for the model
    input_text = f"translate English to SQL: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate SQL query using the model, limiting the token length
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100)  # Adjust token length if necessary

    # Decode the generated SQL query into a readable string
    sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql_query

# Example usage of the function
if __name__ == "__main__":
    input_query = "What is the name of the employee who works in the sales department?"
    sql_result = generate_sql(input_query)
    print(f"Generated SQL query: {sql_result}")
