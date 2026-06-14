"""
Code Sentinel GitHub Action — automated pull-request code review.

Fetches the PR diff from the GitHub API, sends each file hunk to the Code Sentinel
inference service, and posts an aggregated review comment on the pull request.

Required environment variables:
    GITHUB_TOKEN: GitHub API token with ``pull-requests: write``.
    GITHUB_REPOSITORY: ``owner/repo`` slug for the target repository.
    PR_NUMBER: Pull request number to review.
    CODE_SENTINEL_API_URL: Base URL of the Code Sentinel FastAPI service.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests

GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
GITHUB_API_VERSION = "2022-11-28"
REQUEST_TIMEOUT_SECONDS = 120
MAX_COMMENT_CHARS = 65_000

# Map file extensions to Code Sentinel language tags (see training data).
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "py",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

# Filenames to skip entirely (lock files and similar).
SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "go.sum",
        "Cargo.lock",
        "Gemfile.lock",
        "composer.lock",
    }
)

# Path glob patterns for generated or vendored content.
SKIP_PATH_GLOBS: Tuple[str, ...] = (
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/vendor/**",
    "**/*.min.js",
    "**/*.min.css",
    "**/*.map",
    "**/*_generated.*",
    "**/*.pb.go",
    "**/*_pb2.py",
)


@dataclass(frozen=True)
class FileHunk:
    """A single reviewable diff hunk within one changed file."""

    filename: str
    hunk_index: int
    diff: str
    language: str


@dataclass(frozen=True)
class HunkReview:
    """Review text returned for one file hunk."""

    filename: str
    hunk_index: int
    review: str


def _require_env(name: str) -> str:
    """
    Read a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The variable value.

    Raises:
        SystemExit: If the variable is missing or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _github_headers(token: str) -> Dict[str, str]:
    """Build standard GitHub REST API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


def _matches_skip_glob(filename: str, pattern: str) -> bool:
    """Return True if ``filename`` matches a recursive glob ``pattern``."""
    return fnmatch(filename, pattern) or fnmatch(filename.replace("\\", "/"), pattern)


def should_skip_file(filename: str) -> bool:
    """
    Decide whether a changed file should be excluded from automated review.

    Skips binary-only entries (handled upstream), lock files, minified assets,
    and common generated or vendored paths.

    Args:
        filename: Repository-relative path from the GitHub files API.

    Returns:
        True if the file should not be sent to Code Sentinel.
    """
    normalized = filename.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]

    if basename in SKIP_FILENAMES:
        return True
    if basename.endswith(".lock"):
        return True

    for pattern in SKIP_PATH_GLOBS:
        if _matches_skip_glob(normalized, pattern):
            return True

    return False


def detect_language(filename: str) -> str:
    """
    Map a file path to a Code Sentinel language tag.

    Args:
        filename: Repository-relative file path.

    Returns:
        Language tag for the Code Sentinel API, or ``"py"`` as a fallback.
    """
    _, ext = os.path.splitext(filename.lower())
    return EXTENSION_TO_LANGUAGE.get(ext, "py")


def fetch_pull_request_files(
    token: str,
    repository: str,
    pr_number: int,
) -> List[dict]:
    """
    List files changed in a pull request via the GitHub REST API.

    Args:
        token: GitHub API token.
        repository: ``owner/repo`` slug.
        pr_number: Pull request number.

    Returns:
        Raw JSON objects from ``GET /repos/{repo}/pulls/{pr}/files``.

    Raises:
        requests.HTTPError: If the GitHub API request fails.
    """
    url = f"{GITHUB_API_URL}/repos/{repository}/pulls/{pr_number}/files"
    files: List[dict] = []
    page = 1

    while True:
        response = requests.get(
            url,
            headers=_github_headers(token),
            params={"per_page": 100, "page": page},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return files


def split_patch_into_hunks(patch: str) -> List[str]:
    """
    Split a unified diff ``patch`` string into individual hunk bodies.

    GitHub's ``patch`` field omits the ``diff --git`` header but includes one or
    more hunks beginning with ``@@``.

    Args:
        patch: Unified diff fragment for a single file.

    Returns:
        List of hunk strings, each starting with a ``@@`` line.
    """
    if not patch or not patch.strip():
        return []

    hunks: List[str] = []
    current: List[str] = []

    for line in patch.splitlines():
        if line.startswith("@@ ") and current:
            hunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        hunks.append("\n".join(current))

    return hunks


def collect_reviewable_hunks(files: Sequence[dict]) -> List[FileHunk]:
    """
    Convert GitHub PR file records into reviewable hunks.

    Args:
        files: Objects returned by ``fetch_pull_request_files``.

    Returns:
        Flat list of hunks ready for Code Sentinel inference.
    """
    reviewable: List[FileHunk] = []

    for file_info in files:
        filename = file_info.get("filename", "")
        if not filename or should_skip_file(filename):
            continue

        # Binary and oversized changes have no text patch.
        patch = file_info.get("patch")
        if not patch:
            continue

        language = detect_language(filename)
        hunks = split_patch_into_hunks(patch)
        for index, hunk in enumerate(hunks, start=1):
            reviewable.append(
                FileHunk(
                    filename=filename,
                    hunk_index=index,
                    diff=hunk,
                    language=language,
                )
            )

    return reviewable


def request_hunk_review(
    api_base_url: str,
    hunk: FileHunk,
) -> str:
    """
    Call Code Sentinel ``POST /review`` for a single diff hunk.

    Args:
        api_base_url: Service base URL (no trailing path required).
        hunk: Diff hunk and language metadata.

    Returns:
        Generated review text from the API.

    Raises:
        requests.HTTPError: If the inference request fails.
        requests.RequestException: On network or timeout errors.
    """
    review_url = urljoin(api_base_url.rstrip("/") + "/", "review")
    response = requests.post(
        review_url,
        json={"diff": hunk.diff, "lang": hunk.language},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    review = payload.get("review", "").strip()
    if not review:
        return "_No review text returned._"
    return review


def review_all_hunks(
    api_base_url: str,
    hunks: Sequence[FileHunk],
) -> List[HunkReview]:
    """
    Run Code Sentinel on every reviewable hunk.

    Args:
        api_base_url: Code Sentinel service base URL.
        hunks: Hunks collected from the pull request.

    Returns:
        List of per-hunk reviews. Failed hunks include an error note instead of
        raising, so one failure does not block the entire PR comment.
    """
    results: List[HunkReview] = []

    for hunk in hunks:
        try:
            review_text = request_hunk_review(api_base_url, hunk)
        except requests.RequestException as exc:
            review_text = f"_Review request failed: {exc}_"
        results.append(
            HunkReview(
                filename=hunk.filename,
                hunk_index=hunk.hunk_index,
                review=review_text,
            )
        )

    return results


def format_review_comment(
    pr_number: int,
    reviews: Sequence[HunkReview],
    *,
    skipped_count: int,
) -> str:
    """
    Aggregate hunk reviews into a single Markdown PR comment.

    Args:
        pr_number: Pull request number (for the header).
        reviews: Generated reviews per hunk.
        skipped_count: Number of changed files skipped by filters.

    Returns:
        Markdown string suitable for ``POST /issues/{id}/comments``.
    """
    lines = [
        "## Code Sentinel Review",
        "",
        f"Automated review for PR #{pr_number} by "
        "[Code Sentinel](https://github.com/harthikrm/code-sentinel).",
        "",
    ]

    if skipped_count:
        lines.append(f"_{skipped_count} file(s) skipped (binary, lock, or generated paths)._")
        lines.append("")

    if not reviews:
        lines.append("_No reviewable diff hunks found in this pull request._")
        return _truncate_comment("\n".join(lines))

    current_file: Optional[str] = None
    for item in reviews:
        if item.filename != current_file:
            current_file = item.filename
            lines.extend(["", f"### `{item.filename}`", ""])

        if item.hunk_index > 1:
            lines.append(f"**Hunk {item.hunk_index}**")
        lines.append(item.review)
        lines.append("")

    return _truncate_comment("\n".join(lines).strip() + "\n")


def _truncate_comment(body: str) -> str:
    """Trim comment body to GitHub's maximum issue comment size."""
    if len(body) <= MAX_COMMENT_CHARS:
        return body
    suffix = "\n\n_(Comment truncated due to GitHub size limits.)_"
    keep = MAX_COMMENT_CHARS - len(suffix)
    return body[:keep] + suffix


def post_pull_request_comment(
    token: str,
    repository: str,
    pr_number: int,
    body: str,
) -> dict:
    """
    Create an issue comment on a pull request.

    Pull requests are issues in the GitHub API, so the PR number is used as the
    issue number.

    Args:
        token: GitHub API token with ``pull-requests: write``.
        repository: ``owner/repo`` slug.
        pr_number: Pull request number.
        body: Markdown comment body.

    Returns:
        JSON response from the GitHub API.

    Raises:
        requests.HTTPError: If comment creation fails.
    """
    url = f"{GITHUB_API_URL}/repos/{repository}/issues/{pr_number}/comments"
    response = requests.post(
        url,
        headers=_github_headers(token),
        json={"body": body},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def run_review() -> None:
    """
    Execute the full PR review workflow.

    Reads configuration from environment variables, fetches the PR diff, calls
    Code Sentinel for each hunk, and posts an aggregated comment.
    """
    token = _require_env("GITHUB_TOKEN")
    repository = _require_env("GITHUB_REPOSITORY")
    pr_number = int(_require_env("PR_NUMBER"))
    api_base_url = _require_env("CODE_SENTINEL_API_URL")

    print(f"Reviewing {repository} PR #{pr_number}")
    files = fetch_pull_request_files(token, repository, pr_number)

    skipped_files = sum(
        1 for file_info in files if should_skip_file(file_info.get("filename", ""))
    )
    hunks = collect_reviewable_hunks(files)
    print(f"Found {len(hunks)} reviewable hunk(s) across {len(files)} changed file(s)")

    reviews = review_all_hunks(api_base_url, hunks)
    comment_body = format_review_comment(
        pr_number,
        reviews,
        skipped_count=skipped_files,
    )

    result = post_pull_request_comment(token, repository, pr_number, comment_body)
    print(f"Posted review comment: {result.get('html_url', result.get('id'))}")


if __name__ == "__main__":
    run_review()
