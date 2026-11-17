from pathlib import Path
from model.FileModel import FileModel
from ingestion import validator

path = Path('../sample_documents/amir.png')

file_validity : FileModel = validator.validate_file(path)
print(file_validity.is_valid)
print(file_validity.message)

