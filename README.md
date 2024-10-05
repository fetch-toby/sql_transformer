# SQL Transformer using Hugging Face

Hugging Face Transformers is an open-source Python library that provides access to thousands of pre-trained Transformers models for natural language processing (NLP), computer vision, audio tasks etc.

This project demonstrates how to use a [T5 transformer model](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/) to translate natural language queries into SQL statements. It includes two programs:
* `sql_generator`: It translates natural language to SQL using a public pre-trained T5 model.
* `sql_translator`: It uses a secret access token to authenticate and work with private models on Hugging Face to translate a SQL query.

* Create a new secret key with [Hugging Face](https://huggingface.co/)

```
hugging_face_access_token = "<your_hugging_face_access_token>"
```

* Install the required dependencies:
```
$ pip install transformers torch huggingface-hub
```
## References:

1. https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL
2. https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/