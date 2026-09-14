# -*- coding: utf-8 -*-
"""详细解析 PPTX:每页标题概览 + 倒数第2页(slide21)完整结构。"""
import re
import zipfile
from collections import Counter
from pathlib import Path

PPTX = Path(r"d:\Desktop\paper_fuxiang\揭榜挂帅项目年度进展汇报(3).pptx")

with zipfile.ZipFile(PPTX) as z:
    slides = sorted(
        (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
        key=lambda n: int(re.search(r"slide(\d+)", n).group(1)),
    )
    print("====== 每页标题概览 ======")
    for i, s in enumerate(slides, 1):
        xml = z.read(s).decode("utf-8", errors="ignore")
        texts = re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S)
        title = " | ".join(t.strip() for t in texts[:4] if t.strip())
        print(f"第{i}页 ({s}): {title[:100]}")

    print("\n====== 倒数第2页 slide21 结构 ======")
    xml = z.read(slides[-2]).decode("utf-8", errors="ignore")
    tags = re.findall(r"<([a-zA-Z]+:?[a-zA-Z]+)[ />]", xml)
    print("标签统计:", Counter(tags).most_common(20))

    rels_name = "ppt/slides/_rels/slide21.xml.rels"
    if rels_name in z.namelist():
        rels = z.read(rels_name).decode("utf-8", errors="ignore")
        print("\nrels 引用:")
        for m in re.findall(r'Target="([^"]+)"[^>]*Type="([^"]+)"', rels):
            print(" ", m[1].split("/")[-1], "->", m[0])

    print("\n全部 a:t 文本:")
    for t in re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S):
        print(" ", repr(t))
