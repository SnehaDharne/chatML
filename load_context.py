import requests
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
client.get_collections()


n_points = 10
def fetch_sklearn_doc(url):
    response = requests.get(url)
    return response.text


def load_context(model, question):
    svm_url = "https://scikit-learn.org/dev/_sources/modules/svm.rst.txt"
    dt_url = "https://scikit-learn.org/dev/_sources/modules/tree.rst.txt"
    lr_url = "https://scikit-learn.org/dev/_sources/modules/linear_model.rst.txt"

    if model == 'svm':
        url = svm_url
    elif model == 'dt':
        url = dt_url
    elif model == 'lr':
        url = lr_url
        
    doc_text = fetch_sklearn_doc(url)
    raw_chunks = doc_text.split('\n\n')
    client.add(
    collection_name="knowledge-base",
    documents=raw_chunks
    )
    results = client.query(
            collection_name="knowledge-base",
            query_text=str(question),
            limit=n_points,
        )
    print(results)
    context = "\n".join(r.document for r in results)
    
    return context.strip()