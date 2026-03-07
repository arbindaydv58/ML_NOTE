def chunk_text(text, size=600, overlap=120):
    if not text:
        return []

    # Keep paragraph boundaries while enforcing character-based chunk size.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if size <= 0:
        raise ValueError("size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    chunks = []
    buf = []

    for p in paragraphs:
        candidate = " ".join(buf + [p])

        if len(candidate) <= size:
            buf.append(p)
            continue

        if buf:
            chunks.append(" ".join(buf).strip())

        # Build overlap tail based on target overlap characters.
        tail = []
        tail_chars = 0
        for prev in reversed(buf):
            tail.insert(0, prev)
            tail_chars += len(prev) + 1
            if tail_chars >= overlap:
                break

        buf = tail + [p]

    if buf:
        chunks.append(" ".join(buf).strip())

    return [c for c in chunks if len(c) > 50]
