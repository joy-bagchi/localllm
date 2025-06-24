import os
import faiss
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama, OllamaEmbeddings

docs_dir = "docs"
documents = []
for root, dirs, files in os.walk(docs_dir):
    for file in filter(lambda ff: ff.endswith('.txt'), files):
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            print(f"Processing file: {os.path.join(root, file)}")
            content = f.read()
            if content.strip():
                doc = Document(page_content=content, metadata={"source": os.path.join(root, file)})
                documents.append(doc)


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# llm = ChatOllama(model="llama3", temperature=0.1, top_p=0.95)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = OllamaEmbeddings(model="llama3")



index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  # Create a flat index for L2 distance
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
print(vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}))

