import datetime as dt
from typing import Any
import requests
import json
import arxiv
import os
import re
from arxiv import UnexpectedEmptyPageError
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime




base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"

BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"

# Only keep the main research areas we care about
KEEP   = {"cs.CL", "cs.AI", "cs.LG", "cs.IR"}
BLOCKS = {"cs.CV", "eess.AS", "cs.SD", "eess.SP", "q-bio.BM"}


def get_authors(authors, first_author=False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output


def get_label(categories):
    output = str()
    if len(categories) != 1:  
        output = ", ".join(str(c) for c in categories)
    else:
        output = categories[0]
    return output


def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


def extract_last_url(text: str) -> str | None:
    """
    Extract the last URL from the text (if any).
    Mainly used for catching links like "Code is available at https://xxx" in abstracts.
    """
    # Match continuous non-whitespace strings starting with http/https
    urls = re.findall(r"https?://\S+", text)
    if not urls:
        return None
    url = urls[-1]
    # 去掉末尾常见标点
    return url.rstrip(".,;:!?)\"]'")

def iter_results_safe(client, search):
    gen = client.results(search)
    while True:
        try:
            yield next(gen)
        except UnexpectedEmptyPageError as e:
            print(f"[arXiv] empty page, stop paging: {e}")
            break
        except StopIteration:
            break

def get_daily_papers(topic, query, max_results=200):
    """
    Fetch arXiv + PapersWithCode metadata and return them as markdown table rows.
    """
    content: dict[str, str] = {}

    # 1. arxiv client
    client = arxiv.Client(
        page_size=100,    
        delay_seconds=3,  
        num_retries=5    
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # 2. iterate over results
    for res in iter_results_safe(client, search):

        paper_id_full  = res.get_short_id()  
        paper_id       = paper_id_full.split("v")[0]  
        update_time    = res.updated.date()
        paper_title    = res.title
        paper_url      = res.entry_id
        paper_abstract = res.summary.replace("\n", " ")
        # Normalize abstract text for matching graphrag-related keywords
        norm_abs = re.sub(r"[\s\-]", "", paper_abstract.lower())

        cats = res.categories                 # e.g. ['cs.CL', 'cs.LG']

        # Filtering rules:
        # 1) Original logic: focus on categories in KEEP and exclude those in BLOCKS.
        # 2) Relaxed rule: if the abstract contains graphrag / graph-rag / graph rag
        #    (all normalized to "graphrag"), keep the paper even if it is not in KEEP
        #    or falls into BLOCKS.
        has_keep_cat   = any(c in KEEP for c in cats)
        has_block_cat  = any(c in BLOCKS for c in cats)
        has_graphrag_k = "graphrag" in norm_abs

        # Skip if it is not in the target categories and the abstract does not contain graphrag
        if (not has_keep_cat) and (not has_graphrag_k):
            continue
        # Skip if it hits blocked categories and the abstract does not contain graphrag
        if has_block_cat and (not has_graphrag_k):
            continue

        collapsed_abs = make_collapsible(paper_abstract)      
        paper_labels   = ", ".join(cats)

        repo_url = "null"
        try:
            r = requests.get(BASE_URL + paper_id_full, timeout=10).json()
            if r.get("official"):
                repo_url = r["official"]["url"]
        except Exception as e:
            print(f"PwC lookup failed for {paper_id_full}: {e}")

        # If PwC does not provide an official repo, try to extract the last URL
        # from the abstract as a potential code link.
        if repo_url == "null":
            abs_url = extract_last_url(paper_abstract)
            if abs_url:
                repo_url = abs_url

        md_row = (
            f"|**{update_time}**|**{paper_title}**|{paper_labels}| "
            f"{collapsed_abs}|[{paper_id_full}]({paper_url})| "
        )
        md_row += f"**[code]({repo_url})**|" if repo_url != "null" else "null|"

        content[paper_id] = md_row

    return {topic: content}


def make_collapsible(text: str, title: str = "Full Abstract") -> str:
    text = text.replace("|", "\\|")      
    return f"<details><summary>{title}</summary>{text}</details>"

def wrap_old_row(md_row: str) -> str:
    if "<details" in md_row:
        return md_row

    newline = "\n" if md_row.endswith("\n") else ""
    row = md_row.rstrip("\n")  
    cells = row.split("|")
    if len(cells) < 8:         
        return md_row

    cells[4] = make_collapsible(cells[4].strip())

    return "|".join(cells) + newline

def update_json_file(filename, data_all):
    # Initialize with empty dict if file does not exist or is empty
    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read().strip()
        json_data = json.loads(content) if content else {}
    else:
        json_data = {}

    for kw in json_data.values():
        for pid in list(kw.keys()):
            kw[pid] = wrap_old_row(kw[pid])

    for data in data_all:
        for keyword, papers in data.items():
            json_data.setdefault(keyword, {}).update(papers)

    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

def json_to_md(filename, md_filename,
               to_web=False,
               use_title=True,
               use_tc=True,
               show_badge=True):
    """
    Convert the JSON file into a markdown README.

    @param filename: path to the input JSON
    @param md_filename: path to the output markdown file
    """

    DateNow = dt.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-', '.')

    # Initialize with empty dict if file does not exist or is empty
    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)
    else:
        data = {}

    # Clean README.md if it already exists, otherwise create it
    with open(md_filename, "w+") as f:
        pass

    # Write data into README.md
    with open(md_filename, "a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if show_badge == True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

        # Add short description of this repository
        f.write("This repository tracks the latest GraphRAG related papers from arXiv.\n\n")

        if use_title == True:
            f.write("## Updated on " + DateNow + "\n\n")
        else:
            f.write("> Updated on " + DateNow + "\n\n")

        f.write("![Monthly Trend](imgs/trend.png)\n\n")

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # Section header for each topic
            f.write(f"## {keyword}\n\n")

            if use_title == True:
                if to_web == False:
                    f.write("|Date|Title|label|Abstract|PDF|Code|\n" + "|---|---|---|---|---|---|\n")
                else:
                    f.write("| Date | Title | label | Abstract | PDF | Code |\n")
                    f.write("|:---------|:---------------|:-------|:------------------|:------|:------|\n")

            # Sort papers by date
            day_content = sort_papers(day_content)

            for _, v in day_content.items():
                if not v:       
                    continue
                f.write(v.rstrip("\n") + "\n")

            f.write(f"\n")

            # Add: back-to-top link
            top_info = f"#Updated on {DateNow}"
            top_info = top_info.replace(' ', '-').replace('.', '')
            f.write(f"<p align=right>(<a href={top_info}>back to top</a>)</p>\n\n")

        if show_badge == True:
            # Badge definitions
            f.write(
                f"[contributors-shield]: https://img.shields.io/github/contributors/bansky-cl/graphrag-paper-arxiv.svg?style=for-the-badge\n")
            f.write(f"[contributors-url]: https://github.com/bansky-cl/graphrag-paper-arxiv/graphs/contributors\n")
            f.write(
                f"[forks-shield]: https://img.shields.io/github/forks/bansky-cl/graphrag-paper-arxiv.svg?style=for-the-badge\n")
            f.write(f"[forks-url]: https://github.com/bansky-cl/graphrag-paper-arxiv/network/members\n")
            f.write(
                f"[stars-shield]: https://img.shields.io/github/stars/bansky-cl/graphrag-paper-arxiv.svg?style=for-the-badge\n")
            f.write(f"[stars-url]: https://github.com/bansky-cl/graphrag-paper-arxiv/stargazers\n")
            f.write(
                f"[issues-shield]: https://img.shields.io/github/issues/bansky-cl/graphrag-paper-arxiv.svg?style=for-the-badge\n")
            f.write(f"[issues-url]: https://github.com/bansky-cl/graphrag-paper-arxiv/issues\n\n")

    print("finished")

def json_to_trend(json_file: str | Path, img_file: str | Path) -> None:
    json_file = Path(json_file).expanduser().resolve()
    img_file  = Path(img_file).expanduser().resolve()

    # Initialize with empty dict if JSON file does not exist or is empty
    if not json_file.exists():
        data = {}
    else:
        with json_file.open("r", encoding="utf‑8") as f:
            content = f.read().strip()
            if not content:
                data = {}
            else:
                data = json.loads(content)

    counts = Counter()
    for topic_dict in data.values():
        for arxiv_id in topic_dict.keys():
            yymm = arxiv_id[:4]
            year  = 2000 + int(yymm[:2])
            month = int(yymm[2:])
            ym_key = f"{year:04d}-{month:02d}"
            counts[ym_key] += 1

    if not counts:
        print("no data")
        return

    ym_dates = {datetime.strptime(k, "%Y-%m"): k for k in counts}
    sorted_keys = [ym_dates[d] for d in sorted(ym_dates)]
    values = [counts[k] for k in sorted_keys]

    plt.figure(figsize=(9, 4))
    plt.plot(sorted_keys, values, marker="o", linewidth=1, label="Monthly count")

    plt.title("ArXiv Papers per Month")
    # plt.xlabel("Month")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.legend()

    img_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(img_file, dpi=300)
    plt.close()
    print(f"✅ trend save in: {img_file}")


if __name__ == "__main__":

    data_collector = []

    # Keywords for the arXiv search query
    keywords = dict[Any, Any]()
    search_terms = [
        # ---  GraphRAG ---
        "graphrag", "graph-rag", "graph rag",
        "grag", "g-rag",
        
        # --- KG-RAG  ---
        "kgrag", "kg-rag", "kg rag",
        
        # --- Subgraph RAG ---
        "subgraph rag", "sub-graph rag",
        
        # --- Retrieval & Augmentation ---
        "subgraph retrieval", "sub-graph retrieval",
        "retrieving subgraph", 
        "subgraph augmented", "sub-graph augmented",
        "subgraph enhanced",  "sub-graph enhanced",
        
        # Reasoning & Completion ---
        "subgraph reasoning",
        "subgraph completion" 
        ]

    keywords["graphrag"] = " OR ".join([f'ti:"{term}"' for term in search_terms])

    for topic, keyword in keywords.items():
        print("Keyword: " + topic)

        data = get_daily_papers(topic, query=keyword, max_results=200)
        data_collector.append(data)

        print("\n")

    json_file = "docs/arxiv-daily.json"
    img_file = "imgs/trend.png"
    md_file = "README.md"

    update_json_file(json_file, data_collector)
    json_to_trend(json_file, img_file)
    json_to_md(json_file, md_file)