def chunk_text(text, size=600, overlap=120):

    paragraphs = text.split("\n")

    chunks=[]
    buf=[]

    for p in paragraphs:
        if len(" ".join(buf))+len(p) < size:
            buf.append(p)
        else:
            chunks.append(" ".join(buf))
            buf=buf[-3:] + [p]

    if buf:
        chunks.append(" ".join(buf))

    return [c.strip() for c in chunks if len(c)>50]