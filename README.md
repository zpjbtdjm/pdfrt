# pdfrt

A CLI tool that parses PDFs, translates them into Chinese, and generates a summary.

## Usage

Pass one or more PDF files or directories, optionally selecting the backend mode.

```bash
python pdfrt.py <file.pdf|dir> [--mode local|remote]
```

## Configuration

Create a `.env` file in the project root with the following variables.

```env
MODEL_ID=...
API_BASE_URL=...
API_KEY=...
MINERU_API_URL=...
MINERU_API_USER=...
MINERU_API_PASS=...
```

The `MINERU_*` variables point to the remote MinerU service.

## Dependencies

Install the required Python packages.

```bash
pip install requests tqdm colorlog loguru python-dotenv openai
```
