from loaders import load_doc
from splitters import test_splitter
print(load_doc())

loaded_doc = load_doc()
splitted_chunks = test_splitter(loaded_doc)

print("The Size of documents loaded : ", len(loaded_doc))

print("The total number of chunks splitted : ", len(splitted_chunks))