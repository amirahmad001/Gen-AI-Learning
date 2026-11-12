class FileModel:
    format: str
    is_valid: bool
    size_mb: float
    message: str

    def __init__(self,format:str, is_valid: bool, size_mb: float,message:str):
        self.format = format
        self.is_valid = is_valid
        self.size_mb = size_mb
        self.message = message

