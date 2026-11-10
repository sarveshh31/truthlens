"""Builds a small FICTIONAL company PDF used by CI (no copyright issues, always the same text)."""
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("About Vantora Systems", [
        "Vantora Systems is a fictional software company founded in 2009 in the city of Lisborn. "
        "It builds warehouse automation software for mid-sized logistics firms.",
        "The company is led by chief executive officer Marta Quillen, who joined in 2015. "
        "Vantora employs 4,820 people across eleven offices.",
        "Its mission is to make every warehouse run without paper by 2030.",
    ]),
    ("Financial Results 2025", [
        "In 2025 Vantora Systems reported revenue of 812 million euros, an increase of 9 percent over 2024.",
        "Operating margin reached 14.2 percent, and net profit was 71 million euros.",
        "The board proposed a dividend of 0.65 euros per share for the year.",
    ]),
    ("Products", [
        "The flagship product is StockPilot, a platform that tracks pallets in real time using camera sensors.",
        "A second product, RouteLoom, plans delivery routes and reduced average fuel use by 11 percent for pilot customers.",
        "A mobile app called ScanBee lets warehouse staff confirm deliveries with one tap.",
    ]),
    ("Sustainability", [
        "Vantora aims to reach net zero emissions by 2035 and cut its office energy use by 40 percent by 2028.",
        "In 2025 the company planted 25,000 trees through a partnership with the Greenfold Trust.",
        "All data centres used by Vantora run on 92 percent renewable electricity.",
    ]),
    ("People and Culture", [
        "Women hold 38 percent of leadership roles at Vantora Systems.",
        "Employees receive 30 days of paid leave per year and a learning budget of 1,500 euros.",
        "The annual employee satisfaction survey scored 4.3 out of 5 in 2025.",
    ]),
    ("Risks and Outlook", [
        "The main risk is the dependence on three large customers, which together make up 37 percent of revenue.",
        "Cyber security incidents are tracked by a dedicated team of 45 specialists.",
        "For 2026 the company expects revenue growth of 8 to 10 percent and plans to open a new office in the city of Brask.",
    ]),
]


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Vantora Systems Annual Report (fictional)")
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 10))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/vantora_demo.pdf")
    build(target)
    print("Wrote", target)