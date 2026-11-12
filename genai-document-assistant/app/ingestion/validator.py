from pathlib import Path
from model import FileModel
from dotenv import load_dotenv
import os
load_dotenv()

supported_format = os.getenv('SUPPORTED_FORMAT').split(',')
supported_file_size = float(os.getenv('SUPPORTED_SIZE_MB'))

def validate_file_format(filepath:Path):
    file_format = filepath.suffix.lower()
    valid = file_format in supported_format
    return {
        'format' : file_format,
        'file_format_valid':valid
    }
    

def validate_size(filepath:Path):
    max_size__kb = supported_file_size * 1024 *1024
    file_size = filepath.stat().st_size
    return {
        'file_size': file_size,
        'file_size_valid' : file_size <= max_size__kb 
    }




def validate_file(filepath: Path) -> FileModel:

    size_valid = validate_size(filepath)
    format_valid = validate_file_format(filepath)

    is_valid = (
        size_valid['file_size_valid']
        and format_valid['file_format_valid']
    )

    if not format_valid['file_format_valid']:
        message = "Unsupported format"
    elif not size_valid['file_size_valid']:
        message = "Unsupported size"
    else:
        message = "Valid file"

    return FileModel(
        format=format_valid['format'],
        is_valid=is_valid,
        size_mb=round(
            size_valid['file_size'] / (1024 * 1024), 2
        ),
        message=message
    )