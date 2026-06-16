from langchain_text_splitters import CharacterTextSplitter
from loaders import load_doc

#init splitter
splitter = CharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 5
    #seperator = '\n\n'
)

doc = load_doc()

chunks = splitter.split_text(text=doc[0].page_content)
print(type(chunks))
print(f"size of doc : {len(doc)}")
def test_splitter(doc):
    global chunks
    for docmnt in doc:
        new_chunks = splitter.split_text(text= docmnt.page_content)
        #print(f"new chubks: {len(new_chunks)}")
        chunks = chunks + new_chunks
    return chunks

test_splitter(doc)
print(type(chunks),len(chunks))


