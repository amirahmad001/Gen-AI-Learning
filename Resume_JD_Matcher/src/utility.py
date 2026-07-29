def cleanDocuments(documents):
    print(type(documents))
    for document in documents:
        document.page_content = cleanText(document.page_content)
    return documents

def cleanText(page_content):
    page_content = page_content.replace('\n', ' ')
    page_content = page_content.replace('\t', ' ')
    page_content = page_content.replace('\r', ' ')
    page_content = page_content.replace('\f', ' ')
    page_content = page_content.replace('\v', ' ')
    return page_content
    
    
