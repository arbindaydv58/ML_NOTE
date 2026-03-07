from sentence_transformers import CrossEncoder

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def warmup_reranker():
    _get_model()


def rerank(query, docs, k=5):
    if not docs:
        return []

    model = _get_model()

    pairs = [(query, d) for d in docs]
    scores = model.predict(pairs)

    ranked = [d for _, d in sorted(zip(scores, docs), reverse=True)]

    k = max(1, min(k, len(ranked)))
    return ranked[:k]
