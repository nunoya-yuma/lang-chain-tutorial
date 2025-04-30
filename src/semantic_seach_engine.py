from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.runnables import chain

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={
            "source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={
            "source": "mammal-pets-doc"},
    ),
]

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "How many distribution centers does Nike have in the US?"
# )

# print(results[0])

# --- Retriever ---


# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

res = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)


print(res)
