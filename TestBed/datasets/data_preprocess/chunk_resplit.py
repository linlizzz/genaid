import os
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


# Load from api_keys.env
load_dotenv("/scratch/work/zhangl9/genaid/secrets/api_keys.env")

# Make the key available to LangChain and OpenAI internally
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment.")




file_path = "/scratch/work/zhangl9/genaid/TestBed/datasets/Käypä_hoito_flat.jsonl"
target_title = "Kohonnut verenpaine"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if entry["title"] == target_title:
            print(f"\nGuideline ID: {entry['guideline_id']}")
            print(f"\nTitle: {entry['title']}")
            print(f"\nKeywords: {entry['keywords']}")
            print(f"\nPage Content:\n{entry['page_content']}")
            page_content = entry["page_content"]
            break
    else:
        print(f"No entry found with title: {target_title}")


text_splitter = SemanticChunker(OpenAIEmbeddings())

# Split the page content into chunks
# 'breakpoint_threshold_amount'( X \in [0,100] ), 'min_chunk_size'-- used to adjust the chunk sizes
# Any difference between sentences greater than the X percentile is split. The default value for X is 95.0

docs = text_splitter.create_documents([page_content])
print(docs[0].page_content)

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile" # or "standard_deviation" or "interquartile"or "gradient"
)

docs = text_splitter.create_documents([page_content])
print(docs[0].page_content)
print(len(docs))


