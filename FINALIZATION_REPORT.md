# AgentForge finalization report

## Included

- Stable project-root-based PDF and FAISS paths, independent of the shell's current directory.
- Automatic discovery of PDFs in `data/`.
- Cached embedding-model initialization.
- Human-readable PDF page numbers (1-based) in retrieval results.
- Query expansion for common `fx` shorthand referring to Python functions.
- A restricted-AST arithmetic calculator integrated into intent routing and answer generation.
- Lazy Groq client/model initialization so the CLI can start without eagerly constructing provider clients; model-backed requests show a clear missing-key error.
- Retry feedback passed separately from retrieved evidence.
- Explicit source/page citation guidance for document-grounded answers.
- Expanded setup, configuration, privacy, limitations, and testing documentation.
- Offline smoke tests for deterministic routing and calculator behavior.

## Validation performed in the build environment

- All Python files passed AST syntax parsing.
- Project compiled successfully with `compileall` before final test additions.
- Full runtime tests were not executed here because the build environment does not have the project's third-party packages installed (the first missing package was `groq`). Run the documented unittest command after installing `requirements.txt`.

## Before public release

1. Add your own `GROQ_API_KEY` to `.env` locally; never share the `.env` file.
2. Install dependencies and run `python -m unittest discover -s tests -v`.
3. Test one greeting, one arithmetic query, one question answered from a PDF with a source/page, one follow-up using conversation context, and one current-information query.
4. Review traces before sharing them; they may include prompts and retrieved text.
5. Do not describe the application as production-ready or hosted until authentication, multi-user isolation, deployment, and service-level monitoring are implemented.
