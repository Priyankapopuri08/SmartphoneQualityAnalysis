"""
Script to create Camera_Quality_Installation_Guide.docx
Generates a standard formatted DOCX file matching the style of OPVQ_Installation_Guide.docx.
"""

import os
import zipfile

def build_docx(filename):
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
    <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:docDefaults>
        <w:rPrDefault>
            <w:rPr>
                <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
                <w:sz w:val="22"/>
            </w:rPr>
        </w:rPrDefault>
    </w:docDefaults>
</w:styles>"""

    sections = [
        ("Camera Quality Analysis: Installation & Execution Guide", "title"),
        ("Overview", "heading"),
        ("This document provides a step-by-step installation guide for setting up a working Camera Quality Analysis environment matching the methodology and formulas from: 'How to Choose your Pre-owned Smartphone?: A Multi-Dimensional Benchmarking of Performance and Quality' (Popuri & Chakraborty, Section 3.8 and Section 4.4).", "body"),
        ("System Requirements", "heading"),
        ("- Ubuntu 20.04 / 22.04 / 24.04 or WSL (Ubuntu)\n- Python 3.8 or higher\n- Standard Scientific Python Libraries: numpy, scipy, pandas, Pillow", "body"),
        ("Step 1: Update System", "heading"),
        ("Run the following commands:\nsudo apt update && sudo apt upgrade -y", "code"),
        ("Step 2: Install Dependencies", "heading"),
        ("Run the following command:\nsudo apt install -y python3 python3-pip python3-numpy python3-scipy python3-pil python3-pandas", "code"),
        ("Step 3: Create Python Virtual Environment (Optional)", "heading"),
        ("python3 -m venv camera_env\nsource camera_env/bin/activate\npip install --upgrade pip", "code"),
        ("Step 4: Install Required Packages", "heading"),
        ("pip install -r requirements.txt", "code"),
        ("Step 5: Run Unified Camera Quality Pipeline", "heading"),
        ("To replicate the full paper benchmark across all device lineages (Tables 8, 9, and 10):\npython3 camera_quality_pipeline.py", "code"),
        ("Step 6: Evaluate Individual Images", "heading"),
        ("To analyze a single image:\npython3 camera_quality_pipeline.py --image sample_images/daylight_outdoor_2018.jpg --output_json results.json", "code"),
        ("Step 7: Run Modular Branches Independently", "heading"),
        ("Each branch can also be executed in isolation:\n- Branch 01 (BRISQUE): cd branches/01_BRISQUE && python3 run_brisque.py\n- Branch 02 (NIQE): cd branches/02_NIQE && python3 run_niqe.py\n- Branch 03 (IL-NIQE): cd branches/03_IL_NIQE && python3 run_il_niqe.py\n- Branch 04 (Subjective MR - Table 8): cd branches/04_Subjective_MR && python3 run_mr.py\n- Branch 05 (Logistic Mapping Q(x) - Table 9): cd branches/05_Logistic_Mapping && python3 run_logistic.py\n- Branch 06 (Correlation SRCC/PLCC/RMSE - Table 10): cd branches/06_Correlation_Analysis && python3 run_correlation.py", "body"),
        ("Academic Note", "heading"),
        ("In the paper, camera quality analysis is characterized using Mean Rank (MR), 5-parameter Logistic Mapping Q(x), and correlation metrics (SRCC, PLCC, RMSE). Always report mapped perceptual scores Q(x) (scale 1.0 to 5.0, higher is better) alongside raw objective scores as detailed in Section 4.4.", "body")
    ]

    p_xml = []
    for text, kind in sections:
        lines = text.split("\n")
        for line in lines:
            if kind == "title":
                p_xml.append(f"""<w:p>
                    <w:pPr><w:jc w:val="center"/></w:pPr>
                    <w:r><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1F497D"/></w:rPr><w:t>{line}</w:t></w:r>
                </w:p>""")
            elif kind == "heading":
                p_xml.append(f"""<w:p>
                    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
                    <w:r><w:rPr><w:b/><w:sz w:val="26"/><w:color w:val="2E74B5"/></w:rPr><w:t>{line}</w:t></w:r>
                </w:p>""")
            elif kind == "code":
                p_xml.append(f"""<w:p>
                    <w:pPr><w:ind w:left="400"/><w:spacing w:before="60" w:after="60"/></w:pPr>
                    <w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/><w:color w:val="333333"/></w:rPr><w:t>{line}</w:t></w:r>
                </w:p>""")
            else:
                p_xml.append(f"""<w:p>
                    <w:pPr><w:spacing w:before="60" w:after="120"/></w:pPr>
                    <w:r><w:t>{line}</w:t></w:r>
                </w:p>""")

    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        {''.join(p_xml)}
    </w:body>
</w:document>"""

    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/document.xml", doc_xml)
        z.writestr("word/styles.xml", styles)
    print(f"Created DOCX: {filename}")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "Camera_Quality_Installation_Guide.docx")
    build_docx(out_path)
