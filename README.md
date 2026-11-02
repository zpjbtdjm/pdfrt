# pdfrt

A CLI tool that parses PDFs, translates them into Chinese, and generates a summary.

## Installation

From the project directory, install into the `MINERU` conda environment:

```bash
conda activate MINERU
python -m pip install -e .
```

This creates the `pdfrt` command in that environment. Keep this directory in
place while using an editable install. In an environment without MinerU,
install local parsing support with `python -m pip install -e '.[local]'`.

## Usage

Pass one or more PDF files or directories, optionally selecting the backend mode.

```bash
pdfrt <file.pdf|dir> [--mode local|remote] [--tier advanced] [-r | -f | -t | -g | -a]
```

For example, `pdfrt paper.pdf` writes `paper_中文.md` next to the PDF. You can
run the command from any directory when its Python environment is active.
Without activating the environment, use `conda run -n MINERU pdfrt paper.pdf`.

By default (or with `-a`) the full pipeline runs in an isolated temp directory:
parse, format, translate, summarize. Pass `-r`, `-f`, `-t`, or `-g` to run
individual steps; these operate directly on the files next to the PDF.

Local parsing uses MinerU 4.1 with `--tier advanced` by default. Tables are not
included in the output; pass `--table true` to keep them. Remote parsing uses
the official MinerU API, requests `advanced`, and falls back to `standard` when
the service does not offer `advanced`.

## Configuration

Create a `.env` file in the current directory or `~/.config/pdfrt/.env` with
the following variables. An editable installation also reads the project root
`.env`. Environment variables take precedence, followed by the current
directory, user config directory, and project root.

```env
MODEL_ID=...
API_BASE_URL=...
API_KEY=...
MINERU_API_URL=https://mineru.net/api
MINERU_API_TOKEN=...
```

`MINERU_API_URL` points to the official MinerU API (defaults to
`https://mineru.net/api`). `MINERU_API_TOKEN` is the API key used only with
`--mode remote`; `MINERU_API_KEY` is accepted as an alias. Local parsing needs
MinerU 4.1; translation and summary generation need the model variables.
