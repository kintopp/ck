# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ck is a semantic code search tool written in Rust. It combines traditional grep functionality with AI-powered semantic search using local embeddings. The tool is published to crates.io as `ck-search` and includes an MCP server for AI agent integration.

## Essential Commands

### Development Workflow

```bash
# Build the entire workspace
cargo build --workspace

# Build just the CLI (main binary)
cargo build -p ck-search

# Run the CLI from source (development testing)
cargo run -p ck-search -- --help
cargo run -p ck-search -- --sem "query" test_files/

# Run a specific binary
cargo run --bin ck -- --help
```

### Testing

```bash
# Run all tests across workspace
cargo test --workspace

# Test a specific crate
cargo test -p ck-engine
cargo test -p ck-cli

# Run a specific test
cargo test test_name
cargo test test_name -- --nocapture  # with output

# Test with each feature combination (CI requirement)
cargo hack test --each-feature --workspace
```

### Quality Checks (REQUIRED before commit)

Run these in order before ANY commit:

```bash
# 1. Format code (auto-fixes)
cargo fmt --all

# 2. Run linter (must fix all warnings)
cargo clippy --workspace --all-features --all-targets -- -D warnings

# 3. Ensure tests pass
cargo test --workspace
```

### MSRV and CI Validation

```bash
# Check minimum supported Rust version
cargo hack check --each-feature --locked --rust-version --workspace

# Verify lockfile is up-to-date
cargo update --workspace --locked

# Full CI check locally
cargo fmt --all --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace
cargo hack test --each-feature --workspace
```

## Architecture Overview

### Workspace Structure

ck uses a modular Rust workspace with clear separation of concerns:

- **`ck-cli`** → Main binary and MCP server (`ck-search` on crates.io)
  - Entry point: `src/main.rs`
  - Library: `src/lib.rs` (for MCP functionality)
  - CLI parsing, output formatting, MCP protocol implementation
  - Orchestrates all other crates

- **`ck-core`** → Shared types and utilities
  - Common error types (`CkError`, `Result<T>`)
  - Language enum and detection
  - Heatmap rendering utilities
  - Foundation types used across all crates

- **`ck-engine`** → Search orchestration
  - Regex search (traditional grep)
  - Semantic search (embedding-based)
  - Hybrid search (combines both with RRF)
  - Lexical search (BM25)
  - Coordinates between index, embeddings, and ANN

- **`ck-index`** → File indexing and management
  - File discovery and filtering (.gitignore, .ckignore)
  - Sidecar management (`.ck/` directories)
  - Manifest tracking (embeddings metadata)
  - Chunk-level incremental indexing with cache

- **`ck-embed`** → Text embedding providers
  - FastEmbed integration (local models)
  - API backends (future: OpenAI, etc.)
  - Embedding model management

- **`ck-ann`** → Approximate Nearest Neighbor search
  - Vector similarity search
  - Index building and querying
  - Tantivy integration for full-text search

- **`ck-chunk`** → Intelligent code segmentation
  - Tree-sitter parsing for 7+ languages
  - Semantic chunking (functions, classes, etc.)
  - Query-based chunking strategies
  - Token-aware splitting with HuggingFace tokenizers

- **`ck-models`** → Model registry and configuration
  - Embedding model metadata
  - Model selection and validation
  - Configuration management

- **`ck-tui`** → Interactive terminal UI
  - Ratatui-based interface
  - Multi-mode search (semantic, regex, hybrid)
  - File preview with syntax highlighting
  - Config persistence

### Data Flow (Semantic Search)

1. **Indexing Phase** (`ck-index` + `ck-chunk` + `ck-embed`)
   - Discover files (respect .gitignore, .ckignore)
   - Chunk files using tree-sitter (language-aware)
   - Generate embeddings with FastEmbed
   - Store in `.ck/` sidecar with manifest
   - Cache embeddings by chunk hash for incremental updates

2. **Search Phase** (`ck-engine` + `ck-ann`)
   - Embed query using same model
   - ANN search finds similar chunks
   - Tantivy provides lexical search
   - RRF fusion for hybrid mode
   - Extract and format results

3. **Output Phase** (`ck-cli`)
   - Format as grep-compatible, JSON, JSONL, or TUI
   - Syntax highlighting and heatmaps
   - MCP JSON-RPC responses for AI agents

### Key Design Patterns

- **Error handling**: `anyhow::Result` for CLI errors, `CkError` for library errors
- **Async/await**: Tokio runtime for I/O and embedding operations
- **Parallelism**: Rayon for CPU-bound tasks (chunking, indexing)
- **Memory efficiency**: Memory-mapped files (`memmap2`) for large data
- **Incremental updates**: blake3 hashing for chunk-level cache invalidation

## Version Management

### Release Process

1. Update `version` in workspace `Cargo.toml`
2. Update all crate versions (use sed or find/replace)
3. Update `CHANGELOG.md` with release notes
4. Run quality checks (clippy, fmt, test)
5. Commit with message: `chore: Bump version to X.Y.Z`
6. Tag: `git tag X.Y.Z` (NO `v` prefix!)
7. Push: `git push origin main --tags`

### Version Tagging Convention

**CRITICAL**: Tags use format `X.Y.Z` (NO `v` prefix)

```bash
# Correct (current standard since 0.3.8+)
git tag 0.7.1
git tag 0.7.2

# Wrong (deprecated)
git tag v0.7.1  # DO NOT USE
```

Always check existing tags: `git tag --sort=-version:refname`

### CHANGELOG.md Format

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- **Feature name**: User-facing description

### Fixed
- **Bug description**: What was broken and how it's fixed

### Technical
- **Implementation details**: For maintainers
```

## Development Context

### When Working on Features

- **Indexing changes**: Likely touch `ck-index`, `ck-chunk`, `ck-embed`
- **Search algorithms**: Focus on `ck-engine`, `ck-ann`
- **CLI/UX changes**: Modify `ck-cli`, update `--help` and README
- **New languages**: Add to `ck-chunk` (tree-sitter parser) and `ck-core` (Language enum)
- **MCP features**: Implement in `ck-cli/src/lib.rs` (MCP module)

### Cross-Platform Requirements

- Test on Ubuntu, Windows, macOS (CI does this)
- Handle path separators correctly
- Windows has issues with FastEmbed in CI (use `--exclude-features fastembed`)

### Performance Considerations

- Indexing ~1M LOC should take < 2 minutes
- Search queries should be sub-500ms
- Aim for 80-90% cache hit rate on incremental indexing
- Index size typically 1-3x source code size

### Quality Standards (Non-Negotiable)

- Zero clippy warnings
- All code formatted with `cargo fmt`
- All tests passing
- New features require tests
- Update `--help` output for CLI changes
- Update README for user-facing features
- Breaking changes require major version bump

## Common Development Tasks

### Adding a New Language

1. Add tree-sitter grammar to `Cargo.toml` workspace dependencies
2. Add language variant to `ck-core/src/lib.rs` `Language` enum
3. Implement parser in `ck-chunk/src/tree_sitter.rs`
4. Add file extension mapping in `Language::from_extension()`
5. Add tests in `ck-chunk/tests/`
6. Update README language support table

### Adding a New CLI Flag

1. Add to `clap` struct in `ck-cli/src/main.rs`
2. Implement logic in appropriate crate
3. Update `--help` output (automatic from clap)
4. Update README examples
5. Add integration test in `ck-cli/tests/`

### Debugging Index Issues

```bash
# Check index status
cargo run -- --status .

# Inspect file chunking
cargo run -- --inspect src/main.rs

# Force rebuild
cargo run -- --clean .

# View detailed logs
RUST_LOG=debug cargo run -- --sem "query" .
```

### Testing MCP Server

```bash
# Start MCP server
cargo run -- --serve

# Test with Claude Desktop
claude mcp add ck-search -s user -- cargo run --bin ck -- --serve
```

## Notes for AI Agents

- This codebase uses `edition = "2024"` (requires Rust 1.88+)
- When analyzing unexpected behavior, document in `UNEXPECTED.md`
- Index storage is in `.ck/` directories (safe to delete)
- Models cached in `~/.cache/ck/models/` or `%LOCALAPPDATA%\ck\cache\models\`
