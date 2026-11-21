"""Reorder WordprocessingML property children into schema order and fill required attributes.

Usage: python3 fixdocx.py in.docx out.docx
"""
import sys
import zipfile
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
q = lambda t: f"{{{W}}}{t}"

ORDER = {
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
            "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
            "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
            "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
            "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
            "sectPr", "pPrChange"],
    "rPr": ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
            "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
            "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
            "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
             "textDirection", "tcFitText", "vAlign", "hideMark"],
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
              "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar",
              "tblLook", "tblCaption", "tblDescription"],
    "trPr": ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit", "trHeight",
             "tblHeader", "tblCellSpacing", "jc", "hidden"],
    "style": ["name", "aliases", "basedOn", "next", "link", "autoRedefine", "hidden", "uiPriority",
              "semiHidden", "unhideWhenUsed", "qFormat", "locked", "personal", "personalCompose",
              "personalReply", "rsid", "pPr", "rPr", "tblPr", "trPr", "tcPr", "tblStylePr"],
    "settings": ["writeProtection", "view", "zoom", "removePersonalInformation", "removeDateAndTime",
                 "doNotDisplayPageBoundaries", "displayBackgroundShape", "printPostScriptOverText",
                 "printFractionalCharacterWidth", "printFormsData", "embedTrueTypeFonts", "embedSystemFonts",
                 "saveSubsetFonts", "saveFormsData", "mirrorMargins", "alignBordersAndEdges",
                 "bordersDoNotSurroundHeader", "bordersDoNotSurroundFooter", "gutterAtTop", "hideSpellingErrors",
                 "hideGrammaticalErrors", "activeWritingStyle", "proofState", "formsDesign", "attachedTemplate",
                 "linkStyles", "stylePaneFormatFilter", "stylePaneSortMethod", "documentType", "mailMerge",
                 "revisionView", "trackRevisions", "doNotTrackMoves", "doNotTrackFormatting", "documentProtection",
                 "autoFormatOverride", "styleLockTheme", "styleLockQFSet", "defaultTabStop", "autoHyphenation",
                 "consecutiveHyphenLimit", "hyphenationZone", "doNotHyphenateCaps", "showEnvelope",
                 "summaryLength", "clickAndTypeStyle", "defaultTableStyle", "evenAndOddHeaders",
                 "bookFoldRevPrinting", "bookFoldPrinting", "bookFoldPrintingSheets",
                 "drawingGridHorizontalSpacing", "drawingGridVerticalSpacing", "displayHorizontalDrawingGridEvery",
                 "displayVerticalDrawingGridEvery", "doNotUseMarginsForDrawingGridOrigin",
                 "drawingGridHorizontalOrigin", "drawingGridVerticalOrigin", "doNotShadeFormData",
                 "noPunctuationKerning", "characterSpacingControl", "printTwoOnOne", "strictFirstAndLastChars",
                 "noLineBreaksAfter", "noLineBreaksBefore", "savePreviewPicture", "doNotValidateAgainstSchema",
                 "saveInvalidXml", "ignoreMixedContent", "alwaysShowPlaceholderText", "doNotDemarcateInvalidXml",
                 "saveXmlDataOnly", "useXSLTWhenSaving", "saveThroughXslt", "showXMLTags",
                 "alwaysMergeEmptyNamespace", "updateFields", "hdrShapeDefaults", "footnotePr", "endnotePr",
                 "compat", "docVars", "rsids", "mathPr", "attachedSchema", "themeFontLang", "clrSchemeMapping",
                 "doNotIncludeSubdocsInStats", "doNotAutoCompressPictures", "forceUpgrade", "captions",
                 "readModeInkLockDown", "smartTagType", "schemaLibrary", "shapeDefaults", "doNotEmbedSmartTags",
                 "decimalSymbol", "listSeparator"],
    "sectPr": ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar",
               "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols", "formProt", "vAlign", "noEndnote",
               "titlePg", "textDirection", "bidi", "rtlGutter", "docGrid", "printerSettings", "sectPrChange"],
}


def reorder(el, order):
    rank = {q(t): i for i, t in enumerate(order)}
    if "mathPr" in order:
        rank["{http://schemas.openxmlformats.org/officeDocument/2006/math}mathPr"] = order.index("mathPr")
    kids = list(el)
    known = [k for k in kids if k.tag in rank]
    if [k.tag for k in known] == sorted((k.tag for k in known), key=lambda t: rank[t]):
        return
    # stable sort: unknown elements keep their position relative to the end
    keyed = sorted(enumerate(kids), key=lambda p: (rank.get(p[1].tag, len(order)), p[0]))
    for k in kids:
        el.remove(k)
    for _, k in keyed:
        el.append(k)


M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MORDER = {"dPr": ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"],
          "rPr": ["lit", "nor", "scr", "sty", "brk", "aln"]}


def fix_tree(root):
    for name, order in MORDER.items():
        rank = {f"{{{M}}}{t}": i for i, t in enumerate(order)}
        for el in root.iter(f"{{{M}}}{name}"):
            if name == "rPr" and el.find(f"{{{M}}}nor") is not None:
                for x in el.findall(f"{{{M}}}sty") + el.findall(f"{{{M}}}scr"):
                    el.remove(x)
            kids = sorted(enumerate(el), key=lambda p: (rank.get(p[1].tag, 99), p[0]))
            for _, k in kids:
                el.remove(k)
            for _, k in kids:
                el.append(k)
    for el in root.iter(q("rPr"), q("pPr")):
        if el.text and el.text.strip():
            el.text = None
        for c in el:
            if c.tail and c.tail.strip():
                c.tail = None
    for name, order in ORDER.items():
        for el in root.iter(q(name)):
            reorder(el, order)
    # table-style tcPr inside styles also handled by iter above
    for pm in root.iter(q("pgMar")):
        for a in ("header", "footer", "gutter"):
            if pm.get(q(a)) is None:
                pm.set(q(a), "708" if a != "gutter" else "0")
    for ns in root.iter(q("nsid")):
        v = ns.get(q("val"))
        if v and len(v) < 8:
            ns.set(q("val"), v.rjust(8, "0"))


def main(src, dst):
    zin = zipfile.ZipFile(src)
    zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.startswith("word/") and item.filename.endswith(".xml"):
            root = etree.fromstring(data)
            fix_tree(root)
            data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        if item.filename.endswith("/"):
            continue
        zout.writestr(item, data)
    zout.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
