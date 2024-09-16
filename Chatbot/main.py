
import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
file_path = (
"E:\\Ragetoturial\\Chatbot\\gen.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()


text_splitter=RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200)
chunks=text_splitter.split_documents(pages)

chunks = []
for page in pages:
    text = page.page_content  # Assuming page_content contains the text
    text_chunks = text_splitter.split_text(text)
    chunks.extend(text_chunks)

print(chunks)
embedding = MistralAIEmbeddings(api_key="HtwXentowdL4jXVMTk2ldcLMo6jxjeF3")
embedding.model = "mistral-embed"  # or your preferred model if available
docs = embedding.embed_documents(chunks)
print(docs)
