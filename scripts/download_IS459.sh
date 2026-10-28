#!/usr/bin/env bash
#
# Resumable variant of download_IS459.sh. Safe to rerun after a crash or Ctrl+C:
# - partial downloads are continued (wget -c)
# - splits that were fully extracted are skipped, without downloading again
# - interrupted extractions are redone from the zip if it is still on disk
#
# A split counts as done when its marker file exists AND its folder holds every
# file the annotation CSV lists for it.

set -euo pipefail

# Base data directory
DATA_DIR="./data/InsectSet/InsectSet459_Train_Val_Test"
mkdir -p "$DATA_DIR"

ANNOTATION_CSV="${DATA_DIR}/InsectSet459_Train_Val_Test_Annotation.csv"

# Files to download (the CSV must come before the zips, it is used to count files)
FILES=(
  "Train_Val_Test_README.txt"
  "InsectSet459_Train_Val_Test_Annotation.csv"
  "Test.zip"
  "Train.zip"
  "Validation.zip"
)

# Expected MD5s hashes (from Zenodo)
declare -A MD5S=(
  ["Train_Val_Test_README.txt"]="86967777884f969894507d5fd7adb991"
  ["InsectSet459_Train_Val_Test_Annotation.csv"]="a285171a0210fef24b2e25ee08d83213"
  ["Test.zip"]="e2037e90f51b35eb89bb9cea516c0a6d"
  ["Train.zip"]="46f535ca43559bcd87b45d6fab189c5f"
  ["Validation.zip"]="0027a6ae81bd5b06526339efcfa285d5"
)

BASE_URL="https://zenodo.org/records/15675220/files"

# File names the annotation CSV lists for a subset (last column: Train/Validation/Test)
expected_files() {
  awk -F, -v s="$1" 'NR > 1 && $NF == s {print $1}' "$ANNOTATION_CSV" | LC_ALL=C sort
}

expected_count() {
  expected_files "$1" | wc -l
}

# Number of annotated files missing from a subset's folder. Files not in the
# CSV are ignored: Train.zip ships one recording the CSV does not list.
missing_count() {
  comm -23 <(expected_files "$1") \
    <(find "${DATA_DIR}/$1" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | LC_ALL=C sort) \
    | wc -l
}

download_and_verify() {
  local filename="$1"
  local out="${DATA_DIR}/${filename}"

  echo "Downloading $filename..."
  wget -c "${BASE_URL}/${filename}?download=1" -O "$out"

  echo "Verifying MD5 for $filename..."
  local expected="${MD5S[$filename]}"
  local actual
  actual="$(md5sum "$out" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "MD5 mismatch for $filename"
    echo "   expected: $expected"
    echo "   actual:   $actual"
    rm -f "$out"
    exit 1
  fi
  echo "MD5 OK"
}

for FILENAME in "${FILES[@]}"; do
  OUT="${DATA_DIR}/${FILENAME}"
  echo "=== $FILENAME ==="

  if [[ "$FILENAME" != *.zip ]]; then
    # Small files are kept on disk; wget -c is a no-op if they are complete
    download_and_verify "$FILENAME"
    echo "Done: $FILENAME"
    echo
    continue
  fi

  SUBSET="${FILENAME%.zip}"
  MARKER="${DATA_DIR}/.${SUBSET}.extracted"
  EXPECTED_COUNT="$(expected_count "$SUBSET")"
  MISSING_COUNT="$(missing_count "$SUBSET")"

  if [[ "$MISSING_COUNT" -eq 0 ]]; then
    if [[ -f "$MARKER" ]]; then
      echo "Already extracted ($EXPECTED_COUNT files), skipping"
      echo
      continue
    fi
    # The zip is only deleted after a successful unzip, so a complete folder
    # without its zip was fully extracted (e.g. by download_IS459.sh)
    if [[ ! -f "$OUT" ]]; then
      echo "Found complete extraction without marker ($EXPECTED_COUNT files), marking as done"
      touch "$MARKER"
      echo
      continue
    fi
    echo "All files present but extraction was not confirmed, extracting again"
  else
    echo "Missing $MISSING_COUNT of $EXPECTED_COUNT files"
    rm -f "$MARKER"
  fi

  download_and_verify "$FILENAME"

  echo "Unzipping $FILENAME..."
  unzip -q -o "$OUT" -d "$DATA_DIR"

  MISSING_COUNT="$(missing_count "$SUBSET")"
  if [[ "$MISSING_COUNT" -ne 0 ]]; then
    echo "$MISSING_COUNT of $EXPECTED_COUNT annotated files missing for $SUBSET after extraction"
    exit 1
  fi

  # Mark before deleting the zip, so an interruption in between is harmless
  touch "$MARKER"
  rm -f "$OUT"   # delete zip after extraction

  echo "Done: $FILENAME ($EXPECTED_COUNT files)"
  echo
done

echo "All downloads, verifications, and extractions completed!"
