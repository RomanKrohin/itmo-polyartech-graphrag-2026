from concurrent.futures import ThreadPoolExecutor

from assistant.graphrag.extraction import extract_chunk
from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings, make_llm


def main() -> None:
    settings = Settings.from_env()
    chunks = list(reversed(load_chunks(settings.data_dir / "corpus")))

    def extract(chunk):
        return chunk, extract_chunk(chunk, make_llm(settings))

    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, (chunk, result) in enumerate(executor.map(extract, chunks), 1):
            print(
                f"cache {index}/{len(chunks)}: {chunk.chunk_id}, "
                f"entities={len(result.entities)}, relations={len(result.relations)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
