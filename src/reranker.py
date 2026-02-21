from sentence_transformers import CrossEncoder

reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query,docs,k=5):

    pairs=[(query,d) for d in docs]

    scores=reranker.predict(pairs)

    ranked=[d for _,d in sorted(zip(scores,docs),reverse=True)]

    return ranked[:k]