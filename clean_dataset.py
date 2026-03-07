import re
from pathlib import Path

INPUT = Path("data/all_weeks.txt")
OUTPUT = Path("data/clean_ml_notes.txt")

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

text = INPUT.read_text(encoding="utf-8")

# -----------------------------
# 1. Remove FILE/PAGE blocks
# -----------------------------
text = re.sub(r"=+\nFILE:.*?\nPAGE:.*?\n=+\n", "", text)

# -----------------------------
# 2. Remove emails
# -----------------------------
text = re.sub(r"\S+@\S+", "", text)

# -----------------------------
# 3. Remove bullet garbage from OCR
# -----------------------------
# Remove common bullet-like symbols produced by OCR.
text = re.sub(r"[•●▪◦]+", " ", text)

# -----------------------------
# 4. Fix multiple blank lines
# -----------------------------
text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

# -----------------------------
# 5. Remove very short garbage lines
# -----------------------------
clean_lines = []
for line in text.split("\n"):
    if len(line.strip()) < 2:
        continue
    clean_lines.append(line.strip())

text = "\n".join(clean_lines)

# -----------------------------
# 6. Fix spacing
# -----------------------------
text = re.sub(r"[ \t]+", " ", text)

# -----------------------------
# 7. Save clean dataset
# -----------------------------
OUTPUT.write_text(text, encoding="utf-8")

print("CLEAN DATASET SAVED ->", OUTPUT)
