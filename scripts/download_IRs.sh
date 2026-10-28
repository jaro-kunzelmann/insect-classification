#!/usr/bin/env bash

set -euo pipefail

# Base data directory
DATA_DIR="./data/irs/openair"
mkdir -p "$DATA_DIR"

# Files to download
FILES=(
  "koli-national-park-summer.zip"
  "koli-national-park-winter.zip"
  "trollers-gill.zip"
  "tyndall-bruce-monument.zip"
  "gill-heads-mine.zip"
)

# Expected MD5s hashes (from Zenodo)
declare -A MD5S=(
  ["koli-national-park-summer.zip"]="91e43bcff0e31fb7e36be20e69f9cc4d"
  ["koli-national-park-winter.zip"]="c3dc6119b24ce5f68fe1c4225d7643bf"
  ["trollers-gill.zip"]="264dee363a1071143bf2ecf2aab7a63a"
  ["tyndall-bruce-monument.zip"]="564d60f8041f659f0154cd8fbfc172b8"
  ["gill-heads-mine.zip"]="859f5e09f728dba81c0af3eb96196fd5"
)

BASE_URL="https://webfiles.york.ac.uk/OPENAIR/IRs"

for FILENAME in "${FILES[@]}"; do
  OUT="${DATA_DIR}/${FILENAME}"

  echo "=== Downloading: $FILENAME ==="
  BASENAME="${FILENAME%.*}"
  wget -c "${BASE_URL}/${BASENAME}/${FILENAME}" -O "$OUT"

  echo "Verifying MD5 for $FILENAME..."
  EXPECTED="${MD5S[$FILENAME]}"
  ACTUAL="$(md5sum "$OUT" | awk '{print $1}')"
  if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "MD5 mismatch for $FILENAME"
    echo "   expected: $EXPECTED"
    echo "   actual:   $ACTUAL"
    rm -f "$OUT"
    exit 1
  fi
  echo "MD5 OK"

  if [[ "$FILENAME" == *.zip ]]; then
    echo "Unzipping $FILENAME..."
    unzip -o "$OUT" -d "$DATA_DIR"
    rm -f "$OUT"   # delete zip after extraction
  fi

  echo "Done: $FILENAME"
  echo
done

echo "All downloads, verifications, and extractions completed!"
