"""
Prepare a paper reviewer dataset from arXiv.

Downloads recent papers from specified arXiv categories, extracts LaTeX sources,
and creates Harbor task directories with instruction.md + search-papers skill.

Each task directory has the structure:
    <output_dir>/<arxiv_id>/
        instruction.md                              # Review task instructions
        latex/template.tex                           # Main LaTeX source
        .claude/skills/search-papers/SKILL.md        # Search skill test suite
        .claude/skills/search-papers/reference.md    # API reference

Usage:
    uv run examples/train_integrations/harbor/prepare_paper_reviewer_dataset.py \
        --categories cs.CL,cs.CV,cs.LG \
        --num-papers 300 \
        --year-range 2023-2025 \
        --output-dir ~/data/harbor/PaperReviews
"""

import argparse
import io
import os
import re
import shutil
import tarfile
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from textwrap import dedent

# Path to the search-papers skill files (relative to this script)
SKILL_DIR = Path(__file__).parent / "search-papers"
INSTRUCTION_TEMPLATE = Path(__file__).parent / "paper_reviewer_instruction_template.md"

# arXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# arXiv e-print download (LaTeX source)
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

# Namespace for Atom feed parsing
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def fetch_arxiv_metadata(
    categories: list[str],
    max_results: int,
    year_start: int,
    year_end: int,
) -> list[dict]:
    """Fetch paper metadata from arXiv API by category.

    Uses the arXiv search API to find recent papers in the given categories.
    Returns a list of dicts with: arxiv_id, title, authors, abstract, categories, published.
    """
    papers = []
    batch_size = min(100, max_results)  # arXiv API max is 100 per request

    for category in categories:
        collected = 0
        start = 0

        while collected < max_results // len(categories):
            query = f"cat:{category}"
            params = urllib.parse.urlencode({
                "search_query": query,
                "start": start,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            })
            url = f"{ARXIV_API_URL}?{params}"

            print(f"  Fetching {category} (offset={start})...")
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    xml_data = resp.read()
            except Exception as e:
                print(f"  Warning: Failed to fetch {url}: {e}")
                break

            root = ET.fromstring(xml_data)
            entries = root.findall(f"{ATOM_NS}entry")

            if not entries:
                break

            for entry in entries:
                # Extract arxiv ID from the entry id URL
                entry_id = entry.find(f"{ATOM_NS}id").text
                # e.g., "http://arxiv.org/abs/2401.12345v1" -> "2401.12345"
                arxiv_id = entry_id.split("/abs/")[-1]
                # Strip version suffix
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

                published = entry.find(f"{ATOM_NS}published").text[:4]  # Year
                pub_year = int(published)
                if pub_year < year_start or pub_year > year_end:
                    continue

                title = entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " ")
                title = re.sub(r"\s+", " ", title)

                authors = []
                for author in entry.findall(f"{ATOM_NS}author"):
                    name = author.find(f"{ATOM_NS}name").text
                    authors.append(name)

                abstract = entry.find(f"{ATOM_NS}summary").text.strip()

                entry_categories = [
                    c.get("term") for c in entry.findall(f"{ATOM_NS}category")
                ]

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "categories": entry_categories,
                    "year": pub_year,
                })
                collected += 1

                if collected >= max_results // len(categories):
                    break

            start += batch_size
            time.sleep(3)  # Respect arXiv rate limit

    # Deduplicate by arxiv_id
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)

    return unique[:max_results]


def download_latex_source(arxiv_id: str, dest_dir: Path) -> bool:
    """Download and extract LaTeX source for a paper.

    Downloads the e-print tar.gz from arXiv and extracts .tex files
    into dest_dir/latex/. Identifies the main .tex file and copies it
    to dest_dir/latex/template.tex.

    Returns True on success, False on failure.
    """
    latex_dir = dest_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgentReviewer/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  Warning: Failed to download e-print for {arxiv_id}: {e}")
        return False

    # Try to extract as tar archive
    try:
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:*") as tf:
            tex_files = []
            for member in tf.getmembers():
                # Sanitize path
                name = member.name
                if ".." in name or name.startswith("/"):
                    continue
                if member.isfile() and name.endswith(".tex"):
                    target = latex_dir / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src:
                        if src:
                            content = src.read()
                            target.write_bytes(content)
                            tex_files.append((name, content))
                elif member.isfile() and (name.endswith(".bbl") or name.endswith(".bib") or name.endswith(".sty")):
                    # Also extract bibliography and style files
                    target = latex_dir / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src:
                        if src:
                            target.write_bytes(src.read())

            if not tex_files:
                print(f"  Warning: No .tex files found in e-print for {arxiv_id}")
                return False

            # Find the main .tex file (contains \documentclass or \begin{document})
            main_tex = None
            for name, content in tex_files:
                text = content.decode("utf-8", errors="replace")
                if r"\documentclass" in text or r"\begin{document}" in text:
                    main_tex = name
                    break

            if main_tex is None:
                # Fallback: use the largest .tex file
                main_tex = max(tex_files, key=lambda x: len(x[1]))[0]

            # Copy main tex to template.tex
            src_path = latex_dir / main_tex
            template_path = latex_dir / "template.tex"
            if src_path != template_path:
                shutil.copy2(src_path, template_path)

            return True

    except tarfile.TarError:
        # Not a tar file — might be a single tex file (gzipped or plain)
        try:
            import gzip
            text = gzip.decompress(data).decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("utf-8", errors="replace")

        if r"\documentclass" in text or r"\begin{document}" in text:
            (latex_dir / "template.tex").write_text(text)
            return True
        else:
            print(f"  Warning: e-print for {arxiv_id} is not a valid LaTeX source")
            return False


def create_task_directory(
    paper: dict,
    output_dir: Path,
    instruction_template: str,
    skill_dir: Path,
) -> bool:
    """Create a complete task directory for a paper.

    Returns True on success, False on failure.
    """
    arxiv_id = paper["arxiv_id"]
    # Use underscore instead of slash for filesystem safety
    safe_id = arxiv_id.replace("/", "_")
    task_dir = output_dir / safe_id

    # Skip if already exists and has instruction.md
    if (task_dir / "instruction.md").exists() and (task_dir / "latex" / "template.tex").exists():
        return True

    task_dir.mkdir(parents=True, exist_ok=True)

    # Download LaTeX source
    if not download_latex_source(arxiv_id, task_dir):
        # Clean up failed directory
        shutil.rmtree(task_dir, ignore_errors=True)
        return False

    # Create instruction.md from template
    authors_str = ", ".join(paper["authors"][:5])
    if len(paper["authors"]) > 5:
        authors_str += f" et al. ({len(paper['authors'])} authors)"

    paper_metadata = dedent(f"""\
        **Paper Metadata:**
        - **Title:** {paper['title']}
        - **Authors:** {authors_str}
        - **arXiv ID:** {arxiv_id}
        - **Year:** {paper['year']}
        - **Categories:** {', '.join(paper['categories'])}
    """)

    instruction_content = instruction_template.replace("{paper_metadata}", paper_metadata)
    (task_dir / "instruction.md").write_text(instruction_content)

    # Copy search-papers skill files into .claude/skills/search-papers/
    skill_dest = task_dir / ".claude" / "skills" / "search-papers"
    skill_dest.mkdir(parents=True, exist_ok=True)

    for skill_file in ["SKILL.md", "reference.md"]:
        src = skill_dir / skill_file
        if src.exists():
            shutil.copy2(src, skill_dest / skill_file)

    return True


def prepare(
    categories: list[str],
    num_papers: int,
    year_start: int,
    year_end: int,
    output_dir: str,
) -> str:
    """Main preparation pipeline."""
    output_path = Path(os.path.expanduser(output_dir)).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load instruction template
    if not INSTRUCTION_TEMPLATE.exists():
        raise FileNotFoundError(f"Instruction template not found at {INSTRUCTION_TEMPLATE}")
    instruction_template = INSTRUCTION_TEMPLATE.read_text()

    # Verify skill files exist
    if not SKILL_DIR.exists():
        raise FileNotFoundError(f"Search-papers skill directory not found at {SKILL_DIR}")

    # Step 1: Fetch paper metadata from arXiv
    print(f"Fetching metadata for ~{num_papers} papers from categories: {categories}")
    print(f"Year range: {year_start}-{year_end}")
    papers = fetch_arxiv_metadata(categories, num_papers, year_start, year_end)
    print(f"Found {len(papers)} papers")

    if not papers:
        print("No papers found. Check your categories and year range.")
        return str(output_path)

    # Step 2: Download LaTeX sources and create task directories
    print(f"\nDownloading LaTeX sources and creating task directories...")
    success_count = 0
    fail_count = 0

    def _process_paper(paper):
        # Rate limit: 3s between arXiv downloads
        time.sleep(3)
        return create_task_directory(paper, output_path, instruction_template, SKILL_DIR)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_process_paper, p): p for p in papers}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error processing {paper['arxiv_id']}: {e}")
                fail_count += 1

            if (success_count + fail_count) % 10 == 0:
                print(f"  Progress: {success_count} succeeded, {fail_count} failed, "
                      f"{len(papers) - success_count - fail_count} remaining")

    print(f"\nDone! {success_count} task directories created at {output_path}")
    if fail_count > 0:
        print(f"  ({fail_count} papers failed to download)")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare paper reviewer dataset from arXiv"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="cs.CL,cs.CV,cs.LG",
        help="Comma-separated arXiv categories (default: cs.CL,cs.CV,cs.LG)",
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=300,
        help="Number of papers to download (default: 300)",
    )
    parser.add_argument(
        "--year-range",
        type=str,
        default="2023-2025",
        help="Year range as START-END (default: 2023-2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/data/harbor/PaperReviews",
        help="Output directory (default: ~/data/harbor/PaperReviews)",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    year_start, year_end = [int(y) for y in args.year_range.split("-")]

    prepare(
        categories=categories,
        num_papers=args.num_papers,
        year_start=year_start,
        year_end=year_end,
        output_dir=args.output_dir,
    )
