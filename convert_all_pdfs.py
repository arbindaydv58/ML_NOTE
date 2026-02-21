import fitz
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

# MUST point to exe, not folder
pytesseract.pytesseract.tesseract_cmd = r"D:\VW\ML\tesseract.exe"

POPPLER = r"C:\Users\User\Downloads\poppler-25.12.0\Library\bin"

INPUT_DIR = "data"
OUTPUT_FILE = "all_weeks.txt"

all_text=[]

for pdf_path in Path(INPUT_DIR).rglob("*.pdf"):

    print("Reading:",pdf_path)

    try:

        doc=fitz.open(pdf_path)

        for page_num,page in enumerate(doc,1):

            text=page.get_text().strip()

            # OCR if text too small
            if len(text)<80:

                print("OCR fallback:",pdf_path.name,"page",page_num)

                images=convert_from_path(
                    str(pdf_path),   # <- IMPORTANT add str()
                    dpi=150,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=POPPLER
                )

                text=pytesseract.image_to_string(images[0])

            if len(text.strip())<20:
                continue

            block=f"""
==============================
FILE: {pdf_path.name}
PAGE: {page_num}
==============================

{text}
"""

            all_text.append(block)

    except Exception as e:
        print("FAILED:",pdf_path,e)

print("\nSaving combined TXT...")

with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
    f.write("\n".join(all_text))

print("DONE →",OUTPUT_FILE)