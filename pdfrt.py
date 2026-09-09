#!/usr/bin/env python

import asyncio
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
from loguru import logger as loguru_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="INFO")

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

load_dotenv(BASE_DIR / ".env")

MODEL_ID = os.environ["MODEL_ID"]
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

MINERU_API_URL = os.environ["MINERU_API_URL"]
MINERU_API_AUTH = (
    os.environ["MINERU_API_USER"],
    os.environ["MINERU_API_PASS"],
)

TRANSLATE_MODEL = MODEL_ID
GENERATE_MODEL = MODEL_ID
TARGET_LANGUAGE = "中文"
MAX_WORKERS = 16
MAX_CHUNK = 3000
MAX_PARSE_RETRIES = 3
MAX_CONCURRENT_PDFS = 8

DEFAULT_LOCAL_DATA = {
    "backend": "pipeline",
    "parse_method": "auto",
    "lang": "en",
    "formula_enable": "true",
    "table_enable": "true",
}
DEFAULT_REMOTE_DATA = {
    "backend": "hybrid-engine",
    "effort": "high",
    "table_enable": "false",
    "image_analysis": "true",
    "response_format_zip": "true",
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
    import subprocess
    mineru_bin = str(Path(sys.executable).parent / "mineru")
    cmd = [
        mineru_bin,
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "-b", data.get("backend", "pipeline"),
        "-m", data.get("parse_method", "auto"),
        "-l", data.get("lang", "en"),
        "-f", data.get("formula_enable", "true"),
        "-t", data.get("table_enable", "true"),
    ]
    log.info("[%s] Local mineru: %s", pdf_path.stem, " ".join(cmd))
    env = {**os.environ, "MINERU_TOOLS_CONFIG_JSON": str(BASE_DIR / "mineru.json")}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"mineru exit {result.returncode}: {stderr_tail}")


def _recognize_pdf_remote(pdf_path: Path, output_dir: Path, data: dict) -> None:
    name = pdf_path.stem
    with pdf_path.open("rb") as f:
        resp = requests.post(
            f"{MINERU_API_URL}/tasks",
            auth=MINERU_API_AUTH,
            data=data,
            files={"files": (pdf_path.name, f, "application/pdf")},
            timeout=7200,
        )
    resp.raise_for_status()
    task = resp.json()
    log.info("[%s] Remote task created, status=%s", name, task["status"])

    while task["status"] in {"pending", "processing"}:
        time.sleep(2)
        task = requests.get(task["status_url"], auth=MINERU_API_AUTH).json()

    assert task["status"] == "completed", task

    r = requests.get(task["result_url"], auth=MINERU_API_AUTH, timeout=7200)
    r.raise_for_status()
    (output_dir / f"{name}.zip").write_bytes(r.content)
    log.info("[%s] Remote zip downloaded (%d bytes)", name, len(r.content))


def format_dir(directory: Path, name: str) -> None:
    zip_path = directory / f"{name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(directory)
        zip_path.unlink()

    gen_dir = directory / name
    md_candidates = list(gen_dir.rglob(f"{name}.md")) if gen_dir.exists() else []
    source_md = md_candidates[0] if md_candidates else None

    target_md = directory / f"{name}.md"
    if source_md and source_md.exists():
        source_md.replace(target_md)

    img_candidates = list(gen_dir.rglob("images")) if gen_dir.exists() else []
    source_images = next((d for d in img_candidates if d.is_dir()), None)
    target_images = directory / f"{name}_images"
    if source_images and source_images.exists():
        if target_images.exists():
            shutil.rmtree(target_images)
        source_images.replace(target_images)

    if not target_md.exists():
        raise FileNotFoundError(f"Could not find the Markdown file generated by mineru: {target_md}")

    get_hierarchical_title(target_md, target_md)
    replace_image_paths(target_md, target_md, name)

    if gen_dir.exists():
        shutil.rmtree(gen_dir)


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
    replaced_content = re.sub(r"!\[\]\(images/", f"![]({name}_images/", md_content)
    output_file.write_text(replaced_content, encoding="utf-8")


def translate_markdown(work_dir: Path, name: str) -> None:
    input_file = work_dir / f"{name}.md"
    output_file = work_dir / f"{name}_{TARGET_LANGUAGE}.md"
    asyncio.run(translate(input_file, output_file, name))
    input_file.unlink(missing_ok=True)


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

    if args.backend:
        default["backend"] = args.backend

    if mode == "remote":
        if args.effort:
            default["effort"] = args.effort
        if args.table is not None:
            default["table_enable"] = opt(args.table)
        if args.image_analysis is not None:
            default["image_analysis"] = opt(args.image_analysis)
    else:
        if args.method:
            default["parse_method"] = args.method
        if args.lang:
            default["lang"] = args.lang
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


async def process_one_pdf(pdf_path: Path, mode: str, data: dict) -> None:
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
        await asyncio.to_thread(format_dir, work_dir, name)

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


async def run_batch(pdf_paths: list[Path], mode: str, data: dict) -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENT_PDFS)
    pbar = tqdm(total=len(pdf_paths), desc="Batch", unit="file")

    async def run_one(pdf_path: Path) -> None:
        async with sem:
            try:
                await process_one_pdf(pdf_path, mode, data)
            except Exception as e:
                log.error("[%s] Failed: %s", pdf_path.stem, e)
            pbar.update(1)

    await asyncio.gather(*(run_one(p) for p in pdf_paths))
    pbar.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="pdfrt",
        description="PDF parse + translate to Chinese.",
        usage="pdfrt <file.pdf|dir> [<file.pdf|dir> ...] [options]",
    )
    parser.add_argument("paths", nargs="+", help="PDF file(s) or directory(ies)")
    parser.add_argument("--mode", choices=["local", "remote"], default="local",
                        help="mineru backend: local (default) or remote")
    parser.add_argument("--backend", default=None, help="override backend")
    parser.add_argument("--method", default=None, help="parse method (local only)")
    parser.add_argument("--lang", default=None, help="OCR language (local only)")
    parser.add_argument("--formula", type=lambda x: x.lower() == "true", default=None,
                        help="enable formula (local only)")
    parser.add_argument("--table", type=lambda x: x.lower() == "true", default=None,
                        help="enable table")
    parser.add_argument("--effort", default=None, help="parsing effort (remote only)")
    parser.add_argument("--image-analysis", type=lambda x: x.lower() == "true", default=None,
                        help="enable image analysis (remote only)")
    args = parser.parse_args()

    pdf_paths = collect_pdfs(args.paths)
    if not pdf_paths:
        log.error("No PDF files found.")
        sys.exit(1)

    data = build_data(args.mode, args)
    log.info("Batch size: %d  (concurrency: %d, mode: %s, data: %s)",
             len(pdf_paths), MAX_CONCURRENT_PDFS, args.mode, data)
    asyncio.run(run_batch(pdf_paths, args.mode, data))
    log.info("All done.")


if __name__ == "__main__":
    main()
