import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest import ingest_pdf

parser = argparse.ArgumentParser(description="Ingest a PDF into TruthLens")
parser.add_argument("pdf", help="path to the PDF, for example data/capgemini_report.pdf")
parser.add_argument("--max-pages", type=int, default=None, help="only read the first N pages (quick test)")
args = parser.parse_args()

ingest_pdf(args.pdf, args.max_pages)
print("INGESTION DONE")