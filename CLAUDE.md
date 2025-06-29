## Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases (>50 files or >100KB total) that exceed context limits, or when prompted by the use to "use gemini", use `gemini -p` with Google Gemini's massive context window. Use `@` syntax for file inclusion: `@src/main.py` (single file), `@src/` (directory), `@./` (entire project), or `gemini --all_files -p "query"`. Paths are relative to current directory. Optimal for comprehensive security audits (`gemini -p "@src/ @api/ Complete security analysis with specific vulnerabilities and fixes"`), performance analysis (`gemini -p "@src/ @config/ Identify bottlenecks with optimization strategies"`), architecture assessment (`gemini -p "@src/ Analyze patterns and technical debt with actionable recommendations"`), and feature verification (`gemini -p "@src/feature/ @tests/ Verify implementation completeness with gap analysis"`). Always provide business context, request specific actionable recommendations with code examples, and ask for prioritized findings with effort estimates for maximum value.

## Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
  - `uv add package-name --script script.py`
  - `uv remove package-name --script script.py`
