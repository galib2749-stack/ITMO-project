"""Render presentation/presentation.pptx to PNG (one per slide) and PDF using
PowerPoint COM automation (win32com). Run on Windows with MS PowerPoint
installed. Produces presentation/slides_png/slide_XX.png and
presentation/presentation.pdf.
"""
import os
import sys
import time

import win32com.client

PPTX_PATH = os.path.abspath("presentation/presentation.pptx")
PNG_DIR = os.path.abspath("presentation/slides_png")
PDF_PATH = os.path.abspath("presentation/presentation.pdf")

os.makedirs(PNG_DIR, exist_ok=True)

powerpoint = win32com.client.Dispatch("PowerPoint.Application")
powerpoint.Visible = 1

try:
    presentation = powerpoint.Presentations.Open(PPTX_PATH, WithWindow=False)

    # Export PDF
    presentation.SaveAs(PDF_PATH, 32)  # 32 = ppSaveAsPDF
    print(f"Saved PDF: {PDF_PATH}")

    # Export each slide as PNG at a decent resolution
    n_slides = presentation.Slides.Count
    for i in range(1, n_slides + 1):
        out_path = os.path.join(PNG_DIR, f"slide_{i:02d}.png")
        presentation.Slides(i).Export(out_path, "PNG", 1920, 1080)
        print(f"Exported {out_path}")

    presentation.Close()
finally:
    powerpoint.Quit()

print(f"Done. {n_slides} slides rendered.")
