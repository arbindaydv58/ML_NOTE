import re

INPUT = "all_weeks.txt"
OUTPUT = "clean_ml_notes.txt"

with open(INPUT, "r", encoding="utf-8") as f:
    text = f.read()

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
text = text.replace("¢", "")
text = text.replace("●", "")
text = text.replace("e ", "")
text = text.replace("•", "")

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
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(text)

print("CLEAN DATASET SAVED →", OUTPUT)