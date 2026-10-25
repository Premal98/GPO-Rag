"""Builds small PDFs in memory so tests don't need sample files."""


def make_pdf(page_texts):
    """Build a simple PDF. Use None for a page with no text (like a scanned page).

    Text must not contain parentheses or backslashes.
    """
    page_ids = [4 + 2 * i for i in range(len(page_texts))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for pid, text in zip(page_ids, page_texts):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_start = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    return pdf


def page_text(n):
    return f"This is page {n} of the test document."


def long_text(word_count, prefix="word"):
    """Text made of numbered words, e.g. 'word0 word1 word2 ...'."""
    return " ".join(f"{prefix}{i}" for i in range(word_count))
