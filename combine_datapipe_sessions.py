#!/usr/bin/env python3
"""Combine chunked DataPipe session files into one JSON file per session.

This utility is meant for the DataPipe upload pattern used in
`/home/fpt-battery-pipe-example/index.html`, where each uploaded filename looks
like:

    <sessionId>__<checkpoint>__c<checkpointIndex>__p<partIndex>.json

The script groups all matching files by `sessionId`, parses each file as JSON,
and combines the per-file trial arrays back into one ordered list of trials for
that participant.

Key assumptions
---------------
- Each input file is a JSON array of trial objects.
- `sessionId` is the filename prefix before the first `__`.
- `trial_index` is the canonical ordering key across the whole session.
- `checkpointIndex` and `partIndex` are mainly tie-breakers and validation aids.
- Nested JSON values are preserved exactly because files are parsed and emitted
  as JSON objects rather than manipulated as text.

Validation behavior
-------------------
- Fatal errors stop output for that session: unreadable file, invalid JSON,
  top-level non-array payloads, non-object entries, or missing/non-integer
  `trial_index`.
- Logical inconsistencies only emit warnings and still write output, for
  example duplicate trial indices, gaps, overlapping file ranges, or
  `data_checkpoint_ind` mismatches.

Example usage
-------------
    python3 scripts/combine_datapipe_sessions.py
    python3 scripts/combine_datapipe_sessions.py --session-id mo5y00nbfek7soi7
    python3 scripts/combine_datapipe_sessions.py --input-dir data --output-dir data/combined
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FILENAME_PATTERN = re.compile(
    r"^(?P<session_id>.+?)__(?P<checkpoint>.+)__c(?P<checkpoint_index>\d+)__p(?P<part_index>\d+)\.json$"
)


class SessionFatalError(Exception):
    """Raised when a session cannot be safely combined."""


@dataclass(frozen=True)
class ParsedFilename:
    session_id: str
    checkpoint: str
    checkpoint_index: int
    part_index: int
    filename: str


@dataclass(frozen=True)
class SourceFile:
    path: Path
    parsed: ParsedFilename


@dataclass(frozen=True)
class TrialRecord:
    trial: dict[str, Any]
    filename: str
    checkpoint: str
    checkpoint_index: int
    part_index: int
    offset: int


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine chunked DataPipe session JSON files into one JSON file per session."
        ),
        epilog=(
            "Examples:\n"
            "  python3 scripts/combine_datapipe_sessions.py\n"
            "  python3 scripts/combine_datapipe_sessions.py --session-id mo5y00nbfek7soi7\n"
            "  python3 scripts/combine_datapipe_sessions.py --input-dir data --output-dir data/combined"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory containing DataPipe JSON chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/combined",
        help="Directory where combined session files will be written (default: %(default)s).",
    )
    parser.add_argument(
        "--session-id",
        help="Only combine a single session id. Exits non-zero if it is not found.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for output JSON (default: %(default)s).",
    )
    return parser.parse_args()


def parse_filename(path: Path) -> ParsedFilename | None:
    # Keep the checkpoint greedy so names like `experiment__welcome` or
    # `task_0_x__completed` are preserved intact; only the trailing `__cN__pM`
    # suffix is structural.
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None

    return ParsedFilename(
        session_id=match.group("session_id"),
        checkpoint=match.group("checkpoint"),
        checkpoint_index=int(match.group("checkpoint_index")),
        part_index=int(match.group("part_index")),
        filename=path.name,
    )


def discover_session_files(input_dir: Path) -> dict[str, list[SourceFile]]:
    session_files: dict[str, list[SourceFile]] = defaultdict(list)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    for path in sorted(input_dir.glob("*.json")):
        parsed = parse_filename(path)
        if parsed is None:
            warn(
                f"Ignoring file with unexpected name format: {path.name}. "
                "Expected <sessionId>__<checkpoint>__cN__pM.json"
            )
            continue
        session_files[parsed.session_id].append(SourceFile(path=path, parsed=parsed))

    return session_files


def load_source_file(source_file: SourceFile) -> tuple[list[TrialRecord], dict[str, Any]]:
    try:
        with source_file.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise SessionFatalError(
            f"{source_file.parsed.filename}: could not read file ({exc})"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SessionFatalError(
            f"{source_file.parsed.filename}: invalid JSON ({exc})"
        ) from exc

    if not isinstance(payload, list):
        raise SessionFatalError(
            f"{source_file.parsed.filename}: top-level JSON must be an array"
        )

    trial_records: list[TrialRecord] = []
    trial_indices: list[int] = []
    previous_index: int | None = None

    for offset, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise SessionFatalError(
                f"{source_file.parsed.filename}: array entry at offset {offset} is not an object"
            )

        trial_index = entry.get("trial_index")
        if not isinstance(trial_index, int) or isinstance(trial_index, bool):
            raise SessionFatalError(
                f"{source_file.parsed.filename}: trial at offset {offset} is missing an integer trial_index"
            )

        if previous_index is not None and trial_index < previous_index:
            # Validation warning is emitted at the session level, but we retain this summary here.
            pass
        previous_index = trial_index
        trial_indices.append(trial_index)

        trial_records.append(
            TrialRecord(
                trial=entry,
                filename=source_file.parsed.filename,
                checkpoint=source_file.parsed.checkpoint,
                checkpoint_index=source_file.parsed.checkpoint_index,
                part_index=source_file.parsed.part_index,
                offset=offset,
            )
        )

    summary = {
        "filename": source_file.parsed.filename,
        "checkpoint": source_file.parsed.checkpoint,
        "checkpoint_index": source_file.parsed.checkpoint_index,
        "part_index": source_file.parsed.part_index,
        "trial_count": len(trial_records),
        "min_trial_index": min(trial_indices) if trial_indices else None,
        "max_trial_index": max(trial_indices) if trial_indices else None,
        "trial_indices": trial_indices,
    }
    return trial_records, summary


def summarize_duplicates(trial_indices: list[int]) -> str | None:
    duplicates = sorted(index for index, count in Counter(trial_indices).items() if count > 1)
    if not duplicates:
        return None
    preview = ", ".join(str(index) for index in duplicates[:10])
    if len(duplicates) > 10:
        preview += ", ..."
    return f"Duplicate trial_index values detected: {preview}"


def summarize_gaps(trial_indices: list[int]) -> str | None:
    unique_sorted = sorted(set(trial_indices))
    gaps: list[str] = []
    for left, right in zip(unique_sorted, unique_sorted[1:]):
        if right - left > 1:
            gaps.append(f"{left + 1}-{right - 1}")
    if not gaps:
        return None
    preview = ", ".join(gaps[:10])
    if len(gaps) > 10:
        preview += ", ..."
    return f"Gaps in trial_index sequence detected: {preview}"


def validate_session(
    session_id: str, trial_records: list[TrialRecord], source_summaries: list[dict[str, Any]]
) -> list[str]:
    warnings: list[str] = []

    trial_indices = [record.trial["trial_index"] for record in trial_records]

    duplicate_warning = summarize_duplicates(trial_indices)
    if duplicate_warning:
        warnings.append(duplicate_warning)

    gap_warning = summarize_gaps(trial_indices)
    if gap_warning:
        warnings.append(gap_warning)

    for source_summary in source_summaries:
        indices = source_summary["trial_indices"]
        for earlier, later in zip(indices, indices[1:]):
            if later < earlier:
                warnings.append(
                    f"{source_summary['filename']}: non-monotonic trial_index order "
                    f"within file ({earlier} followed by {later})"
                )
                break

    ordered_sources = sorted(
        source_summaries,
        key=lambda item: (item["checkpoint_index"], item["part_index"], item["filename"]),
    )
    previous_source: dict[str, Any] | None = None
    for source_summary in ordered_sources:
        if previous_source is not None:
            prev_max = previous_source["max_trial_index"]
            curr_min = source_summary["min_trial_index"]
            if prev_max is not None and curr_min is not None and curr_min <= prev_max:
                warnings.append(
                    "Overlapping or backward trial ranges across files: "
                    f"{previous_source['filename']} ends at {prev_max}, "
                    f"{source_summary['filename']} starts at {curr_min}"
                )
        previous_source = source_summary

    mismatches: list[str] = []
    for record in trial_records:
        value = record.trial.get("data_checkpoint_ind")
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            mismatches.append(
                f"{record.filename} trial_index {record.trial['trial_index']} has non-integer "
                f"data_checkpoint_ind={value!r}"
            )
            continue
        if value != record.checkpoint_index:
            mismatches.append(
                f"{record.filename} trial_index {record.trial['trial_index']} has "
                f"data_checkpoint_ind={value}, expected {record.checkpoint_index}"
            )
    if mismatches:
        preview = "; ".join(mismatches[:5])
        if len(mismatches) > 5:
            preview += "; ..."
        warnings.append(f"data_checkpoint_ind mismatches detected: {preview}")

    for message in warnings:
        warn(f"{session_id}: {message}")

    return warnings


def build_output_document(
    session_id: str, trial_records: list[TrialRecord], source_summaries: list[dict[str, Any]], warnings: list[str]
) -> dict[str, Any]:
    # `trial_index` is the canonical trial order across the whole participant
    # session. Filename-derived indices are only used as deterministic
    # tie-breakers when trial_index alone is insufficient.
    ordered_trials = sorted(
        trial_records,
        key=lambda record: (
            record.trial["trial_index"],
            record.checkpoint_index,
            record.part_index,
            record.offset,
            record.filename,
        ),
    )

    output_source_files = []
    for summary in sorted(
        source_summaries,
        key=lambda item: (item["checkpoint_index"], item["part_index"], item["filename"]),
    ):
        output_source_files.append(
            {
                "filename": summary["filename"],
                "checkpoint": summary["checkpoint"],
                "checkpoint_index": summary["checkpoint_index"],
                "part_index": summary["part_index"],
                "trial_count": summary["trial_count"],
                "min_trial_index": summary["min_trial_index"],
                "max_trial_index": summary["max_trial_index"],
            }
        )

    return {
        "session_id": session_id,
        "trial_count": len(ordered_trials),
        "source_files": output_source_files,
        "validation_warnings": warnings,
        "trials": [record.trial for record in ordered_trials],
    }


def combine_session(session_id: str, files: list[SourceFile]) -> dict[str, Any]:
    trial_records: list[TrialRecord] = []
    source_summaries: list[dict[str, Any]] = []

    # Read files in checkpoint/part order so range-based validation reflects the
    # expected save sequence from the original experiment run.
    for source_file in sorted(
        files,
        key=lambda item: (
            item.parsed.checkpoint_index,
            item.parsed.part_index,
            item.parsed.filename,
        ),
    ):
        file_records, source_summary = load_source_file(source_file)
        trial_records.extend(file_records)
        source_summaries.append(source_summary)

    warnings = validate_session(session_id, trial_records, source_summaries)
    return build_output_document(session_id, trial_records, source_summaries, warnings)


def write_output(output_dir: Path, session_id: str, document: dict[str, Any], indent: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=indent)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        session_files = discover_session_files(input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.session_id is not None:
        if args.session_id not in session_files:
            print(
                f"Session id not found in {input_dir}: {args.session_id}",
                file=sys.stderr,
            )
            return 1
        target_sessions = {args.session_id: session_files[args.session_id]}
    else:
        target_sessions = dict(sorted(session_files.items()))

    fatal_errors = 0

    for session_id, files in target_sessions.items():
        try:
            document = combine_session(session_id, files)
        except SessionFatalError as exc:
            print(f"Error: {session_id}: {exc}", file=sys.stderr)
            fatal_errors += 1
            continue

        write_output(output_dir, session_id, document, args.indent)
        print(
            f"Wrote {output_dir / f'{session_id}.json'} "
            f"({document['trial_count']} trials from {len(document['source_files'])} files)",
            file=sys.stderr,
        )

    return 1 if fatal_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
