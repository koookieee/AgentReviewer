FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app