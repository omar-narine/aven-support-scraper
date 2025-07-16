from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone
import os
import json
from typing import List

load_dotenv()   

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# TO BE UPDATED
class Metadata(BaseModel):
    description: str
    title: str
    url: str
    sourceURL: str

class SupportContext(BaseModel):
    content: str
    metadata: Metadata
    
class SupportContextList(BaseModel):
    chunks: List[SupportContext]
    
class EmbeddingsDocument(BaseModel):
    content: str
    embedding: list[float]
    metadata: Metadata

class EmbeddingsDocumentList(BaseModel):
    documents: List[EmbeddingsDocument]




# 1 Iterate through the JSON and get Gemini to organize the markdown into a better structured format and perhaps break it down into sections

system_instructions = '''
### Identity
You are an agent who is tasked with organizing content for a company called Aven who provides a home equity credit card. A lot of information has been scraped off of their website which will be used to provide customer support information to customers. Right now the information is in a very dense format and needs to be broken down into smaller sections. 

### Input Data

You will be given a JSON object that contains the following fields:
- `data`: An array of scraped pages, where each page contains:
  - `markdown`: The main content of the page in markdown format
  - `metadata`: Object containing page metadata including:
    - `title`: Page title
    - `description`: Page description
    - `url`: Page URL
    - `sourceURL`: Original source URL
    - `language`: Page language (usually "en")
    - `scrapeId`: Unique identifier for the scrape
    - `statusCode`: HTTP status code
    - `contentType`: Content type of the page
    - `numPages`: Number of pages (for PDFs)
    - Other metadata fields as available

### Instructions

Your task is to process each page in the data array and break down the markdown content into smaller, more manageable chunks while preserving the important metadata. For each page:

1. **Analyze the content structure**: Identify natural sections, topics, or logical breaks in the content
2. **Create focused chunks**: Break down large content into smaller, self-contained pieces that are:
   - Between 100-500 words each (when possible)
   - Focused on a single topic or concept
   - Self-contained and understandable on their own
   - Appropriate for customer support queries

3. **Preserve metadata**: For each chunk, maintain the essential metadata:
   - `title`: Use the original page title or create a descriptive title for the chunk
   - `description`: Create a brief description of what the chunk covers
   - `url`: Use the original page URL
   - `sourceURL`: Use the original sourceURL

4. **Content organization guidelines**:
   - **FAQ sections**: Break into individual Q&A pairs
   - **Feature descriptions**: Separate each feature into its own chunk
   - **Process steps**: Group related steps together
   - **Educational content**: Break into logical sections
   - **Legal/regulatory content**: Keep related information together
   - **Product information**: Separate different product aspects
   - **Contact/support information**: Keep together as one chunk

5. **Quality standards**:
   - Remove navigation menus, footers, and other non-content elements
   - Preserve important formatting (bold, italics, lists)
   - Maintain links and references where relevant
   - Ensure each chunk provides value for customer support

6. **Output format**: Return a JSON object with a "chunks" array containing SupportContext objects, where each object contains:
   - `content`: The chunked content (cleaned and focused)
   - `metadata`: Object with title, description, url, and sourceURL

### Examples of good chunking:

**Before (large content block):**
- A 2000-word article about HELOC rates with multiple sections

**After (chunked):**
- Chunk 1: "What is a HELOC rate?" (200 words)
- Chunk 2: "How are HELOC rates determined?" (300 words)  
- Chunk 3: "Factors that affect your HELOC rate" (250 words)
- Chunk 4: "Comparing HELOC rates to other loan types" (300 words)

### Important Notes:
- Focus on creating chunks that would be useful for answering specific customer questions
- Maintain the original meaning and context of the content
- Remove duplicate or redundant information across chunks
- Ensure each chunk is complete and doesn't leave the reader hanging
- Preserve important legal disclaimers and regulatory information
- Keep related information together when it makes sense for customer support

Process the entire data array and return a comprehensive set of well-structured, focused content chunks that will be effective for customer support purposes.

**IMPORTANT**: Your response must be a JSON object with a "chunks" array containing all the processed content chunks.
'''

with open("crawl_result.json", "r") as f:
    crawl_data = json.dumps(json.load(f))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instructions,
        response_mime_type="application/json",
        response_schema=SupportContextList
    ),
    contents=[
        crawl_data
    ]
)
    
print(response)
response_data = json.loads(response.text)
with open("cleaned_crawl_result.json", "w") as f:
    json.dump(response_data, f, indent=2, ensure_ascii=False)

    

# Converting to Embeddings

response_data = json.load(open("cleaned_crawl_result.json"))

embedding_documents = []

for chunk in response_data["chunks"]:
    embeddings = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["content"],
        config=types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="SEMANTIC_SIMILARITY"
        )
    )
    # Create EmbeddingsDocument instance
    embedding_doc = EmbeddingsDocument(
        content=chunk["content"],
        embedding=embeddings.embeddings[0].values,
        metadata=Metadata(**chunk["metadata"])
    )
    embedding_documents.append(embedding_doc)

# Create EmbeddingsDocumentList
embedding_doc_list = EmbeddingsDocumentList(documents=embedding_documents)

# Write to file
with open("embeddings_output.json", "w") as f:
    f.write(embedding_doc_list.model_dump_json(indent=2))


# Upserting to Pinecone
with open("embeddings_output.json", "r") as f:
    embedding_doc_list = EmbeddingsDocumentList.model_validate_json(f.read())
    
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(host="https://aven-customer-support-as246xh.svc.aped-4627-b74a.pinecone.io")

for i, doc in enumerate(embedding_doc_list.documents):
    index.upsert(
        vectors=[{
            "id": f"doc_{i}",
            "values": doc.embedding,
            "metadata": doc.metadata.model_dump()
        }],
        namespace="aven-scraped-data",
    )






