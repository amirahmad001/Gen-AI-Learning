from langchain_community.document_loaders import Docx2txtLoader
from pathlib import Path

# 1. Define your path object
path = Path('../../sample_documents/Resume.docx')
print(f"File exists: {path.exists()}")

# 2. You can pass the 'path' object directly into the first argument
loader = Docx2txtLoader(file_path=path)

# 3. Process and view the result
documents = loader.load()

print("\n--- First Element ---")
print(documents[0])
