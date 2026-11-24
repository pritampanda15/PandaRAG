"""CLI entrypoint for `pandarag`."""
import argparse
import os
from pathlib import Path
import sys

import uvicorn


def main():
    p = argparse.ArgumentParser(description="Start the PandaRAG glass-box RAG & research workbench.")
    p.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8787")))
    p.add_argument("--plugins", choices=("bio", "core"), default=None, help="Enable full bio labs or core RAG only")
    p.add_argument("--reload", action="store_true", help="Auto-reload on backend edits")
    args = p.parse_args()

    if args.plugins:
        os.environ["PANDARAG_PLUGINS"] = args.plugins

    backend_dir = str(Path(__file__).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload, app_dir=backend_dir)


if __name__ == "__main__":
    main()
