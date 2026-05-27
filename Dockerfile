FROM rust:alpine AS builder

# Install required system dependencies and build tools
RUN apk add --no-cache \
    musl-dev \
    gcc \
    g++ \
    make \
    cmake \
    pkgconfig \
    perl

WORKDIR /usr/src/app

ARG CARGO_FEATURES=""

# Separate dependency cache build from earlier stages
# First copy Cargo.toml and Cargo.lock, as well as resources needed by the build script
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY assets/ ./assets/

# Create a dummy entry file to build the dependency cache
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release ${CARGO_FEATURES} && \
    rm -rf src

# Copy the actual project source code
COPY src ./src

# Ensure the main program is recompiled
RUN touch src/main.rs && cargo build --release ${CARGO_FEATURES}

# Runtime base image: Use Alpine
FROM alpine:latest

RUN apk add --no-cache \
    ca-certificates \
    tzdata

WORKDIR /app

# Copy the compiled binary from the builder image
COPY --from=builder /usr/src/app/target/release/memory-mcp-server /app/memory-mcp-server

# Copy assets needed for runtime (like the ONNX model, etc.)
COPY --from=builder /usr/src/app/assets /app/assets

# Default port is 9180
EXPOSE 9180

# Enable info level logging by default
ENV RUST_LOG=info

# Set the entrypoint
ENTRYPOINT ["/app/memory-mcp-server"]
