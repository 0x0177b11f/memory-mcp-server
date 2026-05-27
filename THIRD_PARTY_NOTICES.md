# Third-Party Notices

This project includes or redistributes third-party software and model assets.

## Model Asset: all-MiniLM-L6-v2

- Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- License: Apache License 2.0
- Local license copy: `assets/all-MiniLM-L6-v2-LICENSE.txt`
- Retrieval and processing:
  - The model originates from the Hugging Face source above.
  - The project uses a processed ONNX artifact (opset 16) tracked in this repository via Git LFS.

### Redistribution Scope

- This repository tracks model-related configuration/tokenizer files under `assets/`.
- The build process uses a processed third-party ONNX model artifact from `assets/model.onnx` at compile time.
- The model payload is embedded into the final binary; runtime binary distributions do not need to include model/tokenizer files from `assets/`.

### Upstream Terms Status

- At the time of review, the upstream model repository page indicates Apache License 2.0.
- No separate additional usage terms were found in the upstream model repository by the maintainer during review.

### Attribution and Compliance Notes

- This project preserves a local copy of the Apache-2.0 license text for the model asset.
- If upstream later adds NOTICE files or additional attribution requirements, this file should be updated accordingly.

### Binary Release Distribution

- Planned distribution includes static binaries built by GitHub Actions CI.
- Binary release archives should include:
  - this third-party notice file;
  - the Apache-2.0 license copy at `assets/all-MiniLM-L6-v2-LICENSE.txt`;
  - clear attribution to the upstream model source.
- Binary release archives are not required to include model/tokenizer runtime files from `assets/`, because the model is embedded in the executable.
