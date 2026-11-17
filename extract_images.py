
from zipfile import ZipFile
from pathlib import Path
from PIL import Image
from io import BytesIO

docx_path = Path(r"D:\Project\dementia-detection-using-deep-learning\Documents\neurofinaldoc.docx")
output_folder = Path(r"D:\Project\dementia-detection-using-deep-learning\figures")

output_folder.mkdir(parents=True, exist_ok=True)

count = 0

with ZipFile(docx_path, "r") as doc:
    for item in doc.namelist():
        if item.startswith("word/media/"):
            data = doc.read(item)
            try:
                img = Image.open(BytesIO(data))
                extension = img.format.lower()
                count += 1
                filename = output_folder / f"project_output_{count:02d}.{extension}"
                filename.write_bytes(data)
                print(f"Extracted: {filename.name}")
            except Exception:
                print(f"Skipped: {item}")

print(f"\nDone! Extracted {count} images.")
print(f"Saved in: {output_folder}")
