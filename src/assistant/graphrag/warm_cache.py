from assistant.graphrag.extraction import extract_chunk
from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings, make_llm


def main() -> None:
    settings = Settings.from_env()
    llm = make_llm(settings)
    chunks = list(reversed(load_chunks(settings.data_dir / "corpus")))
    for index, chunk in enumerate(chunks, 1):
        result = extract_chunk(chunk, llm)
        print(
            f"cache {index}/{len(chunks)}: {chunk.chunk_id}, "
            f"entities={len(result.entities)}, relations={len(result.relations)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
