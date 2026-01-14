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

# Track SSL warnings to avoid spam
_pwc_ssl_warned = False


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
        except arxiv.HTTPError as e:
            # Handle transient arXiv server errors (e.g., HTTP 500) gracefully
            print(f"[arXiv] HTTP error while querying API, stop paging: {e}")
            break
        except StopIteration:
            break

def _load_existing_ids(json_file: str | Path, topic: str) -> set[str]:
    """
    Load existing arXiv IDs for a given topic from the JSON store.
    Stored keys are version-less IDs like "2401.01234".
    """
    json_path = Path(json_file).expanduser()
    if not json_path.exists():
        return set()
    try:
        content = json_path.read_text(encoding="utf-8").strip()
        if not content:
            return set()
        data = json.loads(content)
        topic_dict = data.get(topic, {}) if isinstance(data, dict) else {}
        return set(topic_dict.keys()) if isinstance(topic_dict, dict) else set()
    except Exception as e:
        print(f"[warn] failed to load existing ids from {json_path}: {e}")
        return set()


def _build_date_range_query(query: str, start_date: str = "2024-01-01") -> str:
    """
    Add an arXiv submittedDate range filter to an existing query.
    arXiv API format: submittedDate:[YYYYMMDD0000 TO *]
    """
    if start_date is None:
        return query
    ymd = start_date.replace("-", "")
    if not (len(ymd) == 8 and ymd.isdigit()):
        raise ValueError(f"start_date must be YYYY-MM-DD, got: {start_date}")
    date_filter = f"submittedDate:[{ymd}0000 TO *]"
    return f"({query}) AND {date_filter}"


def get_daily_papers(
    topic,
    query,
    max_results=500,
    *,
    json_file: str | Path | None = None,
    start_date: str = "2024-01-01",
):
    """Fetch arXiv + PapersWithCode metadata and return them as markdown rows."""
    content: dict[str, str] = {}
    existing_ids: set[str] = set()
    if json_file is not None:
        existing_ids = _load_existing_ids(json_file, topic)

    total_seen = 0
    total_kept = 0
    total_new = 0

    client = arxiv.Client(
        page_size=100,    
        delay_seconds=3,  
        num_retries=5    
    )

    # Single query (no batching)
    query_with_date = _build_date_range_query(query, start_date=start_date)

    search = arxiv.Search(
        query=query_with_date,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for res in iter_results_safe(client, search):
        total_seen += 1

        paper_id_full  = res.get_short_id()  
        paper_id       = paper_id_full.split("v")[0]  
        update_time    = res.updated.date()
        paper_title    = res.title
        paper_url      = res.entry_id
        paper_abstract = res.summary.replace("\n", " ")
        norm_abs = re.sub(r"[\s\-]", "", paper_abstract.lower())

        cats = res.categories
        has_keep_cat   = any(c in KEEP for c in cats)
        has_block_cat  = any(c in BLOCKS for c in cats)
        has_graphrag_k = "graphrag" in norm_abs

        if (not has_keep_cat) and (not has_graphrag_k):
            continue
        if has_block_cat and (not has_graphrag_k):
            continue

        total_kept += 1
        if paper_id not in existing_ids:
            total_new += 1

        collapsed_abs = make_collapsible(paper_abstract)
        # Markdown 表格中不允许原始的 `|`，否则会打乱列对齐
        safe_title     = str(paper_title).replace("|", "\\|")
        paper_labels   = ", ".join(cats)

        repo_url = "null"
        global _pwc_ssl_warned
        if not _pwc_ssl_warned:
            try:
                r = requests.get(BASE_URL + paper_id_full, timeout=5).json()
                if r.get("official"):
                    repo_url = r["official"]["url"]
            except requests.exceptions.SSLError as e:
                print(f"[PwC] SSL handshake failed in this environment, will skip PwC lookups. Example: {e}")
                _pwc_ssl_warned = True
            except Exception as e:
                print(f"PwC lookup failed for {paper_id_full}: {e}")

        if repo_url == "null":
            abs_url = extract_last_url(paper_abstract)
            if abs_url:
                repo_url = abs_url

        md_row = (
            f"|**{update_time}**|**{safe_title}**|{paper_labels}| "
            f"{collapsed_abs}|[{paper_id_full}]({paper_url})| "
        )
        md_row += f"**[code]({repo_url})**|" if repo_url != "null" else "null|"

        content[paper_id] = md_row

    batches_info = ""
    if json_file is None:
        print(
            f"[{topic}] arXiv fetched={total_seen} (max_results={max_results}, since={start_date}){batches_info}; kept={total_kept}"
        )
    else:
        print(
            f"[{topic}] arXiv fetched={total_seen} (max_results={max_results}, since={start_date}){batches_info}; "
            f"kept={total_kept}; new={total_new}; already_have={len(existing_ids)}"
        )

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
               show_badge=True,
               show_label_and_abstract=True):
    """
    Convert the JSON file into a markdown README.
    """

    DateNow = dt.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-', '.')

    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)
    else:
        data = {}

    with open(md_filename, "w+") as f:
        pass

    with open(md_filename, "a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if show_badge == True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

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
            f.write(f"## {keyword}\n\n")

            if use_title == True:
                if to_web == False:
                    if show_label_and_abstract:
                        f.write("|Date|Title|label|Abstract|PDF|Code|\n" + "|---|---|---|---|---|---|\n")
                    else:
                        f.write("|Date|Title|PDF|Code|\n" + "|---|---|---|---|\n")
                else:
                    if show_label_and_abstract:
                        f.write("| Date | Title | label | Abstract | PDF | Code |\n")
                        f.write("|:---------|:---------------|:-------|:------------------|:------|:------|\n")
                    else:
                        f.write("| Date | Title | PDF | Code |\n")
                        f.write("|:---------|:---------------|:------|:------|\n")

            day_content = sort_papers(day_content)

            for _, v in day_content.items():
                if not v:       
                    continue
                row = v.rstrip("\n")
                if not show_label_and_abstract:
                    # 原始行为 6 列: |Date|Title|label|Abstract|PDF|Code|
                    cells = row.split("|")
                    if len(cells) >= 7:
                        # cells 结构: ["", date, title, label, abstract, pdf, code, ""]
                        kept = [cells[1], cells[2], cells[5], cells[6]]
                        row = "|" + "|".join(kept) + "|"
                    # 防止旧数据里摘要（<details>...</details>）被挤进 Title 等单元格
                    row = re.sub(r"<details>.*?</details>", "", row, flags=re.DOTALL)
                f.write(row + "\n")

            f.write(f"\n")

            top_info = f"#Updated on {DateNow}"
            top_info = top_info.replace(' ', '-').replace('.', '')
            f.write(f"<p align=right>(<a href={top_info}>back to top</a>)</p>\n\n")

        if show_badge == True:
            f.write(
                f"[contributors-shield]: https://img.shields.io/github/contributors/bansky-cl/graphrag-arxiv-daily-paper.svg?style=for-the-badge\n")
            f.write(f"[contributors-url]: https://github.com/bansky-cl/graphrag-arxiv-daily-paper/graphs/contributors\n")
            f.write(
                f"[forks-shield]: https://img.shields.io/github/forks/bansky-cl/graphrag-arxiv-daily-paper.svg?style=for-the-badge\n")
            f.write(f"[forks-url]: https://github.com/bansky-cl/graphrag-arxiv-daily-paper/network/members\n")
            f.write(
                f"[stars-shield]: https://img.shields.io/github/stars/bansky-cl/graphrag-arxiv-daily-paper.svg?style=for-the-badge\n")
            f.write(f"[stars-url]: https://github.com/bansky-cl/graphrag-arxiv-daily-paper/stargazers\n")
            f.write(
                f"[issues-shield]: https://img.shields.io/github/issues/bansky-cl/graphrag-arxiv-daily-paper.svg?style=for-the-badge\n")
            f.write(f"[issues-url]: https://github.com/bansky-cl/graphrag-arxiv-daily-paper/issues\n\n")

    print("finished")

def json_to_trend(json_file: str | Path, img_file: str | Path) -> None:
    json_file = Path(json_file).expanduser().resolve()
    img_file  = Path(img_file).expanduser().resolve()

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
    keywords = dict[Any, Any]()
    search_terms = [
        # ---  GraphRAG (core terms) ---
        "graphrag", "graph-rag", "graph rag",
        "grag", "g-rag",

        # --- KG-RAG  ---
        "kgrag", "kg-rag", "kg rag",

        # --- Subgraph RAG / Subgraph-level methods ---
        "subgraph rag", "sub-graph rag",
        "subgraph-level rag", "sub-graph-level rag",

        # --- Hybrid graph-text / hybrid retrieval ---
        "hybrid rag", "hybrid graph rag",
        "hybrid retrieval-augmented generation",
        "hybrid graph retrieval",

        # --- Retrieval & Augmentation around subgraphs / graphs ---
        "subgraph retrieval", "sub-graph retrieval",
        "retrieving subgraph",
        "graph retrieval-augmented generation",
        "graph-augmented generation", "graph augmented generation",
        "graph-enhanced generation", "graph enhanced generation",

        # --- Enhancement / augmentation wording for subgraphs / graphs ---
        "subgraph augmented", "sub-graph augmented",
        "subgraph enhanced",  "sub-graph enhanced",
        "graph augmented", "graph-augmented",
        "graph enhanced",  "graph-enhanced",

        # --- Reasoning & Completion ---
        "subgraph reasoning", "graph reasoning",
        "subgraph completion", "graph completion",
        ]

    keywords["graphrag"] = " OR ".join([f'ti:"{term}"' for term in search_terms])

    # Output files
    json_file = "docs/arxiv-daily.json"
    img_file = "imgs/trend.png"
    md_file = "README.md"

    json_path = Path(json_file)
    if json_path.exists():
        content = json_path.read_text(encoding="utf-8").strip()
        if not content:
            is_bootstrap = True
        else:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    has_any_papers = any(isinstance(v, dict) and len(v) > 0 for v in data.values())
                    is_bootstrap = not has_any_papers
                else:
                    is_bootstrap = False
            except Exception:
                is_bootstrap = False
    else:
        is_bootstrap = True

    fetch_start_date = None
    fetch_max_results = 500 if is_bootstrap else 200

    for topic, keyword in keywords.items():
        print("Keyword: " + topic)

        data = get_daily_papers(
            topic,
            query=keyword,
            max_results=fetch_max_results,
            json_file=json_file,
            start_date=fetch_start_date,
        )
        data_collector.append(data)

        print("\n")

    update_json_file(json_file, data_collector)
    json_to_trend(json_file, img_file)
    # README size limit, so we don't show label and abstract
    json_to_md(json_file, md_file, show_label_and_abstract=False)