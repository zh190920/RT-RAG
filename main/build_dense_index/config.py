# config.py

# Path to the directory containing the original raw dataset JSON files
raw_path = "/path/to/your/raw/data"

# Path where processed chunks, FAISS index, and config files will be saved
save_path = "/path/to/save/embedding/results"

# Base URL of the OpenAI-compatible API endpoint
base_url = "/path/to/your/base/url"

# Your OpenAI API key (keep this secure)
api_key = "your-api-key"

# Name of the dataset file (without .json extension)
dataset_name = "2wikimultihopqa"#2wikimultihopqa/hotpotqamusique

# Maximum number of words per chunk
chunk_size = 200

# Minimum number of sentences required in each chunk
min_sentence = 2

# Number of overlapping sentences between consecutive chunks
overlap = 2
