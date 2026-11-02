#!/usr/bin/env python

import asyncio
import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import colorlog

_handler = colorlog.StreamHandler()
_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s%(reset)s [%(thin_cyan)s%(levelname)s%(reset)s] %(message_log_color)s%(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
    secondary_log_colors={
        "message": {
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    },
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("pdfrt")

BASE_DIR = Path(__file__).resolve().parent

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

for env_file in (
    Path.cwd() / ".env",
    Path.home() / ".config" / "pdfrt" / ".env",
    BASE_DIR / ".env",
):
    load_dotenv(env_file)

MODEL_ID = os.environ.get("MODEL_ID", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

MINERU_API_URL = os.environ.get("MINERU_API_URL", "https://mineru.net/api")
MINERU_API_TOKEN = os.environ.get("MINERU_API_TOKEN", "") or os.environ.get("MINERU_API_KEY", "")

TRANSLATE_MODEL = MODEL_ID
GENERATE_MODEL = MODEL_ID
TARGET_LANGUAGE = "中文"
MAX_WORKERS = 16
MAX_CHUNK = 3000
MAX_PARSE_RETRIES = 3
MAX_CONCURRENT_PDFS = 8

DEFAULT_TIER = "advanced"
TABLE_BLOCK_TYPES = {"table", "table_body", "table_caption", "table_footnote"}

DEFAULT_LOCAL_DATA = {
    "tier": DEFAULT_TIER,
    "parse_method": "auto",
    "formula_enable": "true",
    "table_enable": "false",
}
DEFAULT_REMOTE_DATA = {
    "tier": DEFAULT_TIER,
    "ocr_mode": "auto",
    "table_enable": "false",
    "output_formats": ["markdown", "zip"],
}


def recognize_pdf(pdf_path: Path, output_dir: Path, mode: str, data: dict) -> None:
    last_err = None
    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        try:
            if mode == "remote":
                _recognize_pdf_remote(pdf_path, output_dir, data)
            else:
                _recognize_pdf_local(pdf_path, output_dir, data)
            return
        except Exception as e:
            last_err = e
            log.warning("[%s] Parse attempt %d/%d failed: %s", pdf_path.stem, attempt, MAX_PARSE_RETRIES, e)
            if attempt < MAX_PARSE_RETRIES:
                time.sleep(5)
    raise RuntimeError(f"PDF parsing failed after {MAX_PARSE_RETRIES} attempts: {last_err}")


def _recognize_pdf_local(pdf_path: Path, output_dir: Path, data: dict) -> None:
    try:
        from mineru.parser import parse
    except ImportError as exc:
        raise RuntimeError("mineru is not installed; install pdfrt[local] in this Python environment") from exc

    tier = data.get("tier", DEFAULT_TIER)
    log.info("[%s] Local mineru parse (tier=%s, method=%s)", pdf_path.stem, tier, data.get("parse_method", "auto"))
    result = parse(
        pdf_path,
        tier=tier,
        ocr_mode=data.get("parse_method", "auto"),
        image_analysis=True,
    )
    _save_parse_result(result, pdf_path, output_dir, data)


def _recognize_pdf_remote(pdf_path: Path, output_dir: Path, data: dict) -> None:
    if not MINERU_API_TOKEN:
        raise RuntimeError("MINERU_API_TOKEN (or MINERU_API_KEY) is required for remote parsing")

    name = pdf_path.stem
    headers = {"Authorization": f"Bearer {MINERU_API_TOKEN}"}
    file_bytes = pdf_path.read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()

    resp = requests.post(
        f"{MINERU_API_URL}/v1/uploads",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "filename": pdf_path.name,
            "bytes": len(file_bytes),
            "mime_type": "application/pdf",
            "purpose": "parse",
            "sha256sum": sha256,
        },
        timeout=300,
    )
    resp.raise_for_status()
    upload = resp.json()
    log.info("[%s] Remote upload session %s (%s)", name, upload["id"], upload["status"])

    if upload.get("status") != "completed":
        resp = requests.put(
            upload["upload_url"],
            data=file_bytes,
            headers=upload.get("upload_headers") or {},
            timeout=7200,
        )
        resp.raise_for_status()

        resp = requests.post(
            f"{MINERU_API_URL}/v1/uploads/{upload['id']}/complete",
            headers={**headers, "Content-Type": "application/json"},
            json={"sha256sum": sha256},
            timeout=300,
        )
        resp.raise_for_status()
        upload = resp.json()

    file_id = upload["file"]["id"]
    payload = {
        "files": [{"source": {"type": "file_id", "file_id": file_id}}],
        "tier": data.get("tier", DEFAULT_TIER),
        "ocr_mode": data.get("ocr_mode", "auto"),
        "output_formats": data.get("output_formats", ["markdown", "zip"]),
    }
    job = _create_remote_job(headers, payload, name)
    job_id = job["job_id"]

    delay = 2.0
    while job["status"] not in {"completed", "partial", "failed", "canceled"}:
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)
        resp = requests.get(
            f"{MINERU_API_URL}/v1/parse/jobs/{job_id}",
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        job = resp.json()
        log.info("[%s] Remote job %s: %s", name, job_id, job["status"])

    if job["status"] != "completed":
        err = (job.get("files") or [{}])[0].get("error")
        raise RuntimeError(f"mineru job {job_id} ended as {job['status']}: {err}")

    outputs = job["files"][0].get("output_files") or {}
    zip_file_id = (outputs.get("zip") or {}).get("file_id")
    if not zip_file_id:
        raise RuntimeError(f"mineru job {job_id} returned no zip output")

    r = requests.get(
        f"{MINERU_API_URL}/v1/files/{zip_file_id}/content",
        headers=headers,
        allow_redirects=True,
        timeout=7200,
    )
    r.raise_for_status()
    (output_dir / f"{name}.zip").write_bytes(r.content)
    log.info("[%s] Remote zip downloaded (%d bytes)", name, len(r.content))


def _create_remote_job(headers: dict, payload: dict, name: str) -> dict:
    resp = requests.post(
        f"{MINERU_API_URL}/v1/parse/jobs",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    if resp.status_code >= 400:
        try:
            error = (resp.json() or {}).get("error") or {}
        except ValueError:
            error = {}
        if error.get("code") == "quality_tier_unavailable" and payload.get("tier") != "standard":
            log.warning("[%s] Remote tier %s unavailable; falling back to standard", name, payload.get("tier"))
            payload = {**payload, "tier": "standard"}
            resp = requests.post(
                f"{MINERU_API_URL}/v1/parse/jobs",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
    resp.raise_for_status()
    job = resp.json()
    log.info("[%s] Remote job %s created (tier=%s)", name, job["job_id"], job.get("tier"))
    return job


def _drop_tables(result) -> None:
    for page in result.middle_json.pages:
        page.blocks = [block for block in page.blocks if str(block.type) not in TABLE_BLOCK_TYPES]


def _materialize_assets(result, pdf_path: Path):
    from docvortex.document.pdf import PDFDocument
    from docvortex.export import materialize_middle

    if pdf_path.suffix.lower() != ".pdf":
        return materialize_middle(result.middle_json)

    with PDFDocument(str(pdf_path)) as document:
        def resolve_image(block, page_idx: int):
            bbox = getattr(block, "bbox", None)
            if bbox is None:
                return None
            try:
                artifact = document.render_image(
                    page_idx,
                    bbox=tuple(float(value) for value in bbox),
                    image_format="png",
                )
            except Exception as exc:
                log.debug("[%s] Skip image at page %d: %s", pdf_path.stem, page_idx + 1, exc)
                return None
            return artifact.data, artifact.extension

        return materialize_middle(result.middle_json, image_resolver=resolve_image)


def _save_parse_result(result, pdf_path: Path, output_dir: Path, data: dict) -> None:
    from mineru.parser import ParseResult

    if str(data.get("table_enable", "false")).lower() != "true":
        _drop_tables(result)

    middle_json, assets = _materialize_assets(result, pdf_path)
    method = data.get("parse_method") or data.get("ocr_mode") or "auto"
    gen_dir = output_dir / pdf_path.stem / method
    gen_dir.mkdir(parents=True, exist_ok=True)

    markdown = ParseResult(middle_json=middle_json).markdown()
    (gen_dir / f"{pdf_path.stem}.md").write_text(markdown, encoding="utf-8")
    for relative_path, payload in assets.items():
        target = gen_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    log.info("[%s] Parsed markdown written (%d image asset(s))", pdf_path.stem, len(assets))


def remove_markdown_tables(text: str) -> str:
    text = re.sub(r"<table\b.*?</table>", "", text, flags=re.IGNORECASE | re.DOTALL)
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    separator = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    while index < len(lines):
        line = lines[index]
        if (
            "|" in line
            and index + 1 < len(lines)
            and separator.match(lines[index + 1])
        ):
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                index += 1
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def format_dir(directory: Path, name: str, table_enable: bool = False) -> None:
    zip_path = directory / f"{name}.zip"
    extracted: set[Path] = set()
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            zf.extractall(directory)
        zip_path.unlink()
        extracted = {directory / m.rstrip("/") for m in members}

    gen_dir = directory / name

    # markdown: {name}/<method>/{name}.md (local mineru) or full.md (official remote zip)
    md_candidates = list(gen_dir.rglob(f"{name}.md")) if gen_dir.exists() else []
    if not md_candidates and (directory / "full.md").exists():
        md_candidates = [directory / "full.md"]
    source_md = md_candidates[0] if md_candidates else None

    # images: {name}/<method>/images (local mineru) or images/ (official remote zip)
    img_candidates = list(gen_dir.rglob("images")) if gen_dir.exists() else []
    source_images = next((d for d in img_candidates if d.is_dir()), None)
    if source_images is None and (directory / "images").is_dir():
        source_images = directory / "images"

    target_md = directory / f"{name}.md"
    if source_md and source_md.exists():
        source_md.replace(target_md)

    target_images = directory / f"{name}_images"
    if source_images and source_images.exists():
        if target_images.exists():
            shutil.rmtree(target_images)
        source_images.replace(target_images)

    if not target_md.exists():
        raise FileNotFoundError(f"Could not find the Markdown file generated by mineru: {target_md}")

    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    for entry in extracted:
        if entry.is_file():
            entry.unlink(missing_ok=True)

    if not table_enable:
        target_md.write_text(remove_markdown_tables(target_md.read_text(encoding="utf-8")), encoding="utf-8")

    get_hierarchical_title(target_md, target_md)
    replace_image_paths(target_md, target_md, name)


def extract_titles_from_markdown(text: str) -> list[str]:
    titles = re.findall(r"^#+\s*(.+?)\s*$", text, flags=re.MULTILINE)
    seen = set()
    return [title for title in titles if not (title in seen or seen.add(title))]


def get_hierarchical_title(input_file: Path, output_file: Path) -> None:
    file_text = input_file.read_text(encoding="utf-8")
    extracted_titles = extract_titles_from_markdown(file_text)
    if not extracted_titles:
        output_file.write_text(file_text.replace("&", ""), encoding="utf-8")
        return

    for _ in range(3):
        markdown_hierarchy = get_hierarchy(extracted_titles)
        if not markdown_hierarchy:
            time.sleep(2)
            continue
        try:
            updated_text = replace_titles_in_text(file_text, extracted_titles, markdown_hierarchy)
            output_file.write_text(updated_text.replace("&", ""), encoding="utf-8")
            return
        except ValueError:
            time.sleep(2)

    output_file.write_text(file_text.replace("&", ""), encoding="utf-8")


def get_hierarchy(titles: list[str]) -> str | None:
    system_prompt = """你是一个Markdown格式转换专家，需要完成以下任务：
1. 分析标题的层级关系（通过标题编号和语义判断层级）
2. 将原始标题转换为标准的Markdown标题格式
3. 保持原始标题顺序，仅添加正确的#符号数量"""
    user_prompt = f"""请将以下标题列表转换为标准Markdown层级格式：
原始标题列表：
{chr(10).join(titles)}
转换规则：
- 主标题使用单个#
- 一级标题使用##
- 二级标题使用###
- 三级标题使用####
- 四级标题使用#####
- 五级标题使用######
- 注意：摘要总是使用##
- 保持原有编号格式不变
- 不要修改标题文字内容
示例转换：
输入：["Attention is all you need", "Abstract", "1 Introduction", "2.1 Background"]
输出：
# Attention is all you need
## Abstract
## 1 Introduction
### 2.1 Background

现在请直接输出转换结果："""
    return chat_complete(system_prompt, user_prompt, TRANSLATE_MODEL, temperature=0.1)


def replace_titles_in_text(text: str, original_titles: list[str], new_hierarchy: str) -> str:
    new_titles = [title for title in new_hierarchy.strip().split("\n") if title.strip()]
    if len(original_titles) != len(new_titles):
        raise ValueError("Original title count does not match generated title count.")

    text = re.sub(r"#+", "#", text)
    for original_title, new_title in zip(original_titles, new_titles):
        text = text.replace(f"# {original_title}", new_title)
    return text


def replace_image_paths(input_file: Path, output_file: Path, name: str) -> None:
    md_content = input_file.read_text(encoding="utf-8")
    replaced_content = re.sub(r"!\[([^\]]*)\]\(images/", rf"![\1]({name}_images/", md_content)
    output_file.write_text(replaced_content, encoding="utf-8")


def merge_chunks(chunks: list[tuple[str, str]], threshold: int = MAX_CHUNK) -> list[tuple[str, str]]:
    merged_chunks = []
    current_content = ""

    for chunk_type, content in chunks:
        if chunk_type == "text":
            if len(current_content) + len(content) >= threshold:
                merged_chunks.append(("text", current_content))
                current_content = content
            else:
                current_content += "\n" + content
        else:
            if current_content:
                merged_chunks.append(("text", current_content))
                current_content = ""
            merged_chunks.append((chunk_type, content))

    if current_content:
        merged_chunks.append(("text", current_content))
    return merged_chunks


async def translate(input_file: Path, output_file: Path, name: str = "") -> None:
    content = input_file.read_text(encoding="utf-8")
    frontmatter_content, body = split_frontmatter(content)

    chunks = []
    for line in body.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith(("```", "~~~")):
            chunks.append(("code", line))
        elif line_stripped.startswith(("<table>", "<html>")):
            chunks.append(("html", line))
        else:
            chunks.append(("text", line))

    chunks = merge_chunks(chunks)
    translated_chunks: list[str | None] = [None] * len(chunks)
    tasks = []
    for index, (chunk_type, chunk_content) in enumerate(chunks):
        if chunk_type in {"text", "html"} and chunk_content.strip():
            tasks.append((index, chunk_type, chunk_content))
        else:
            translated_chunks[index] = chunk_content

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    sem = asyncio.Semaphore(MAX_WORKERS)
    desc = f"Translate [{name}]" if name else "Translating"
    pbar = tqdm(total=len(tasks), desc=desc, unit="chunk", leave=False)

    async def translate_one(idx: int, ctype: str, ccontent: str) -> tuple[int, str]:
        async with sem:
            result = await process_chunk(client, ctype, ccontent)
            pbar.update(1)
            return idx, result

    results = await asyncio.gather(*(translate_one(i, t, c) for i, t, c in tasks))
    for idx, result in results:
        translated_chunks[idx] = result
    pbar.close()

    translated_body = "\n".join(chunk or "" for chunk in translated_chunks)
    output_file.write_text(frontmatter_content + "\n" + translated_body, encoding="utf-8")


def split_frontmatter(content: str) -> tuple[str, str]:
    if not content.startswith("---"):
        return "", content
    end_index = content.find("---", 3)
    if end_index == -1:
        return "", content
    return content[: end_index + 3], content[end_index + 3 :].lstrip("\n")


async def process_chunk(client: AsyncOpenAI, chunk_type: str, content: str) -> str:
    for retry_count in range(2):
        try:
            if chunk_type == "text":
                prompt = f"请将以下内容翻译为{TARGET_LANGUAGE}。仅输出翻译结果，不要输入任何要求之外的内容，严格保持Markdown格式：\n\n{content}"
                result = await chat_complete_async(client, "你是一个优秀的翻译专家", prompt, TRANSLATE_MODEL, temperature=0)
            elif chunk_type == "html":
                prompt = f"请将以下html格式转换为Markdown，并严格保持Markdown格式输出：\n\n{content}"
                result = await chat_complete_async(client, "你是一个优秀的格式转换专家", prompt, TRANSLATE_MODEL, temperature=0)
            else:
                return content
            return remove_specific_markers(result or content)
        except Exception:
            if retry_count == 1:
                return content
            await asyncio.sleep(3)
    return content


def remove_specific_markers(text: str) -> str:
    return text.replace("```markdown", "").replace("```", "")


def get_summary(directory: Path, name: str) -> None:
    input_file = directory / f"{name}_{TARGET_LANGUAGE}.md"
    if not input_file.exists():
        raise FileNotFoundError(f"File does not exist: {input_file}")

    text = input_file.read_text(encoding="utf-8")
    result = generate_summary(text)
    insert_summary_into_file(input_file, result)


def generate_summary(text: str) -> str:
    prompt = f"""
内容文本：
{text}
请基于以上内容文本，总结文章的主要内容。
"""
    return stream_and_collect("", prompt, GENERATE_MODEL)


def insert_summary_into_file(file_path: Path, summary: str) -> None:
    content = file_path.read_text(encoding="utf-8")
    quoted_summary = format_as_blockquote(summary)
    match = re.search(r"^# .+$", content, re.MULTILINE)
    if match:
        insert_pos = match.end()
        new_content = content[:insert_pos] + "\n\n" + quoted_summary + "\n" + content[insert_pos:]
    else:
        new_content = quoted_summary + "\n\n" + content
    file_path.write_text(new_content, encoding="utf-8")


def format_as_blockquote(text: str) -> str:
    lines = text.strip().split("\n")
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


async def chat_complete_async(
    client: AsyncOpenAI, system_prompt: str, user_prompt: str, model: str, temperature: float
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""


def chat_complete(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""


def stream_and_collect(system_prompt: str, user_prompt: str, model: str) -> str:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        temperature=0.1,
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    return result + "\n"


def build_data(mode: str, args) -> dict:
    default = dict(DEFAULT_REMOTE_DATA if mode == "remote" else DEFAULT_LOCAL_DATA)

    def opt(val):
        return str(val).lower() if isinstance(val, bool) else val

    if args.tier:
        default["tier"] = args.tier

    if mode == "remote":
        if args.method:
            default["ocr_mode"] = args.method
        if args.table is not None:
            default["table_enable"] = opt(args.table)
        return default

    if args.method:
        default["parse_method"] = args.method
    if args.formula is not None:
        default["formula_enable"] = opt(args.formula)
    if args.table is not None:
        default["table_enable"] = opt(args.table)
    return default


def collect_pdfs(paths: list[str]) -> list[Path]:
    pdfs = []
    for arg in paths:
        p = Path(arg).expanduser().resolve()
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            log.warning("Skip (not a PDF or directory): %s", arg)
    return pdfs


ALL_STEPS = ("recognize", "format", "translate", "generate")


def resolve_steps(args) -> list[str]:
    steps = [step for step in ALL_STEPS if getattr(args, step)]
    if args.all or not steps:
        return list(ALL_STEPS)
    return steps


async def run_full_pipeline(pdf_path: Path, mode: str, data: dict) -> None:
    name = pdf_path.stem

    dest_dir = pdf_path.parent
    md_out = dest_dir / f"{name}_中文.md"
    img_out = dest_dir / f"{name}_images"

    if md_out.exists():
        log.info("[%s] Output exists, skip.", name)
        return

    work_dir = Path(tempfile.mkdtemp(prefix=f"pdfrt_{name}_"))
    try:
        log.info("[%s] [1/4] Parsing PDF (mode=%s) ...", name, mode)
        await asyncio.to_thread(recognize_pdf, pdf_path, work_dir, mode, data)

        log.info("[%s] [2/4] Formatting Markdown ...", name)
        await asyncio.to_thread(format_dir, work_dir, name, str(data.get("table_enable", "false")).lower() == "true")

        log.info("[%s] [3/4] Translating to Chinese ...", name)
        await translate(work_dir / f"{name}.md", work_dir / f"{name}_{TARGET_LANGUAGE}.md", name)
        (work_dir / f"{name}.md").unlink(missing_ok=True)

        log.info("[%s] [4/4] Generating summary ...", name)
        await asyncio.to_thread(get_summary, work_dir, name)

        src_md = work_dir / f"{name}_中文.md"
        md_out.write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")

        src_img = work_dir / f"{name}_images"
        if src_img.exists():
            if img_out.exists():
                shutil.rmtree(img_out)
            shutil.copytree(src_img, img_out)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    log.info("[%s] Done: %s", name, md_out)


async def process_one_pdf(pdf_path: Path, mode: str, data: dict, steps: list[str]) -> None:
    if steps == list(ALL_STEPS):
        await run_full_pipeline(pdf_path, mode, data)
        return

    name = pdf_path.stem
    dest_dir = pdf_path.parent

    if "recognize" in steps:
        log.info("[%s] Recognizing PDF (mode=%s) ...", name, mode)
        await asyncio.to_thread(recognize_pdf, pdf_path, dest_dir, mode, data)

    if "format" in steps:
        log.info("[%s] Formatting Markdown ...", name)
        await asyncio.to_thread(format_dir, dest_dir, name, str(data.get("table_enable", "false")).lower() == "true")

    if "translate" in steps:
        log.info("[%s] Translating to Chinese ...", name)
        await translate(dest_dir / f"{name}.md", dest_dir / f"{name}_{TARGET_LANGUAGE}.md", name)
        (dest_dir / f"{name}.md").unlink(missing_ok=True)

    if "generate" in steps:
        log.info("[%s] Generating summary ...", name)
        await asyncio.to_thread(get_summary, dest_dir, name)

    log.info("[%s] Steps done: %s", name, "+".join(steps))


async def run_batch(pdf_paths: list[Path], mode: str, data: dict, steps: list[str]) -> bool:
    sem = asyncio.Semaphore(MAX_CONCURRENT_PDFS)
    pbar = tqdm(total=len(pdf_paths), desc="Batch", unit="file")

    async def run_one(pdf_path: Path) -> bool:
        async with sem:
            try:
                await process_one_pdf(pdf_path, mode, data, steps)
                return True
            except Exception as e:
                log.error("[%s] Failed: %s", pdf_path.stem, e)
                return False
            finally:
                pbar.update(1)

    results = await asyncio.gather(*(run_one(p) for p in pdf_paths))
    pbar.close()
    return all(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="pdfrt",
        description="PDF parse + translate to Chinese.",
        usage="pdfrt <file.pdf|dir> [<file.pdf|dir> ...] [options]",
    )
    parser.add_argument("paths", nargs="+", help="PDF file(s) or directory(ies)")
    parser.add_argument("--recognize", "-r", action="store_true",
                        help="run only the PDF recognition step")
    parser.add_argument("--format", "-f", action="store_true",
                        help="run only the Markdown formatting step")
    parser.add_argument("--translate", "-t", action="store_true",
                        help="run only the translation step")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="run only the summary generation step")
    parser.add_argument("--all", "-a", action="store_true",
                        help="run all steps (default when no step flag is given)")
    parser.add_argument("--mode", choices=["local", "remote"], default="local",
                        help="mineru backend: local (default) or remote")
    parser.add_argument("--tier", choices=["flash", "basic", "standard", "advanced"], default=None,
                        help="mineru parse tier (default: advanced)")
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default=None,
                        help="parse method (local) or OCR mode (remote)")
    parser.add_argument("--formula", type=lambda x: x.lower() == "true", default=None,
                        help="enable formula (kept for compatibility)")
    parser.add_argument("--table", type=lambda x: x.lower() == "true", default=None,
                        help="enable table parsing (default: false)")
    args = parser.parse_args()

    steps = resolve_steps(args)
    pdf_paths = collect_pdfs(args.paths)
    if not pdf_paths:
        log.error("No PDF files found.")
        sys.exit(1)

    required = []
    if "translate" in steps or "generate" in steps:
        required.extend(("MODEL_ID", "API_BASE_URL", "API_KEY"))
    missing = [name for name in required if not os.environ.get(name)]
    if args.mode == "remote" and "recognize" in steps and not MINERU_API_TOKEN:
        missing.append("MINERU_API_TOKEN")
    if missing:
        parser.error(f"missing configuration: {', '.join(missing)}")

    data = build_data(args.mode, args)
    log.info("Batch size: %d  (concurrency: %d, mode: %s, steps: %s, data: %s)",
             len(pdf_paths), MAX_CONCURRENT_PDFS, args.mode, "+".join(steps), data)
    ok = asyncio.run(run_batch(pdf_paths, args.mode, data, steps))
    if ok:
        log.info("All done.")
    if args.mode == "local" and "recognize" in steps:
        try:
            from mineru.model.vlm.runtime import shutdown_cached_models
            shutdown_cached_models()
        except Exception as exc:
            log.debug("MinerU runtime shutdown skipped: %s", exc)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if ok else 1)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
