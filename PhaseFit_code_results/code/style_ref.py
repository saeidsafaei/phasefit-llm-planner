"""Create the pandoc reference document (styles) for the manuscript."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

d = Document("/home/claude/paper/out/ref_default.docx")
st = d.styles
BYNAME = {s.name: s for s in st}
def S(name):
    return BYNAME[name]
FONT = "Times New Roman"

def setfont(s, size, bold=None, italic=None, color=(0, 0, 0)):
    s.font.name = FONT
    s.font.size = Pt(size)
    if bold is not None:
        s.font.bold = bold
    if italic is not None:
        s.font.italic = italic
    s.font.color.rgb = RGBColor(*color)
    rpr = s.element.get_or_add_rPr()
    rfonts = rpr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        if rfonts.get(W + a) is not None:
            del rfonts.attrib[W + a]
    for a in ("ascii", "hAnsi", "cs", "eastAsia"):
        rfonts.set(W + a, FONT)

for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
    if name in BYNAME:
        s = S(name)
        setfont(s, 12)
        pf = s.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        if name != "Compact":
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for name, size, b, i in (("Heading 1", 14, True, False), ("Heading 2", 12, True, False),
                         ("Heading 3", 12, False, True), ("Title", 16, True, False),
                         ("Subtitle", 12, False, True)):
    s = S(name)
    setfont(s, size, b, i)
    s.paragraph_format.space_before = Pt(12 if name.startswith("Heading") else 0)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.keep_with_next = True
    if name in ("Title", "Subtitle"):
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
for name in ("Image Caption", "Table Caption", "Caption"):
    if name in BYNAME:
        s = S(name); setfont(s, 10, False, False)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        s.paragraph_format.space_after = Pt(10)
for name in ("Abstract",):
    if name in BYNAME:
        s = S(name); setfont(s, 11)
for name in ("Hyperlink",):
    if name in BYNAME:
        S(name).font.color.rgb = RGBColor(0x1F, 0x4E, 0x9A)
# custom paragraph styles used through fenced divs
for nm, size, b, i, al in (("Authors", 12, False, False, WD_ALIGN_PARAGRAPH.CENTER),
                           ("Affiliation", 10, False, True, WD_ALIGN_PARAGRAPH.CENTER),
                           ("Keywords", 11, False, False, WD_ALIGN_PARAGRAPH.LEFT),
                           ("Reference", 10, False, False, WD_ALIGN_PARAGRAPH.LEFT),
                           ("TableNote", 9, False, False, WD_ALIGN_PARAGRAPH.LEFT),
                           ("TableTitle", 10, False, False, WD_ALIGN_PARAGRAPH.LEFT)):
    if nm not in [s.name for s in st]:
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = S("Normal")
        BYNAME[nm] = s
    s = BYNAME[nm]; setfont(s, size, b, i)
    s.paragraph_format.alignment = al
    s.paragraph_format.line_spacing = 1.15 if nm in ("Reference", "TableNote", "TableTitle") else 1.5
    s.paragraph_format.space_after = Pt(4 if nm == "Reference" else 6)
    if nm == "TableTitle":
        s.paragraph_format.keep_with_next = True
        s.paragraph_format.space_before = Pt(10)
# page setup
sec = d.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)
d.save("/home/claude/paper/out/reference.docx")
print("ok")
