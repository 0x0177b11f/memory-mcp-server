# Changelog

## Unreleased

- No changes yet.

## 0.2.0

### Added

- Added hybrid search parameters to `list_documents` and updated the related database schema.
- Added a document update tool, with tests and documentation.
- Added minimum-distance filtering support to memory search.
- Added a migrate-only startup mode.

### Changed

- Optimized document and memory database queries.
- Refactored search SQL builder logic and expanded database retrieval tests.
- Refactored database module layout for clearer separation of operations.
- Updated server tool result serialization to JSON instead of debug-style strings.
- Updated package version code to `0.2.0`.
- Linux release packaging is defined for `x86_64-unknown-linux-musl` GitHub Releases.
- Release archives include the project license, third-party notices, and the model license copy.

### Fixed

- Fixed Docker image build configuration in `Dockerfile`.
- Fixed `.dockerignore` to avoid incorrect build context contents.
- Fixed GitHub Actions Linux release pipeline dependency setup.

### Documentation

- Updated `README.md` for newly added capabilities and operational flows.

## 0.1.0

### Fist Release

- Document collection management (create, list, delete)
- Memory chunk operations (insert, delete)
- Similarity search by content
- Similarity search by summary
- Combined similarity search across summary and content
- Optional metadata filtering for search results
