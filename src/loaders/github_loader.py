"""
src/loaders/github_loader.py

Loads historical bug data from the GitHub REST API.

Handles two types of GitHub projects:
  - Projects that use formal Issue Assignees   (e.g. microsoft/vscode)
  - Projects that use Pull Requests only        (e.g. facebook/react, ungoogled-chromium)

For PR-based projects, the PR *author* is treated as the developer who fixed the issue.

Returns
-------
bugs        : list[Bug]   — resolved issues/PRs with a known developer.
dev_stats   : dict        — aggregated developer metrics (input to base.build_developers).
"""

import datetime
import json
import urllib.request
import urllib.error
from pathlib import Path

from src.models.bug import Bug
from src.loaders.base import map_severity

_GITHUB_API = "https://api.github.com"
_HEADERS = {"User-Agent": "BugClassifier/2.0", "Accept": "application/vnd.github+json"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(repo: str, limit: int = 100) -> tuple[list[Bug], dict]:
    """
    Fetch the most recently closed issues and PRs from a GitHub repository.

    Parameters
    ----------
    repo  : "owner/repo" string, e.g. "microsoft/vscode".
    limit : Maximum number of items to fetch (GitHub API max per page is 100).

    Returns
    -------
    (bugs, dev_stats)
    """
    raw_items = _fetch_closed_items(repo, limit)
    bugs, dev_stats = _process_items(raw_items)
    print(f"[GitHub Loader] {repo}: {len(bugs)} resolved items, "
          f"{len(dev_stats)} unique developers")
    return bugs, dev_stats


def fetch_single_issue(repo: str, issue_number: str | int) -> dict | None:
    """
    Fetch the metadata for a single GitHub issue by number.

    Parameters
    ----------
    repo         : "owner/repo" string.
    issue_number : The issue number shown in the GitHub URL.

    Returns
    -------
    Raw JSON dict from the GitHub API, or None on failure.
    """
    url = f"{_GITHUB_API}/repos/{repo}/issues/{issue_number}"
    return _get(url)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_closed_items(repo: str, limit: int) -> list[dict]:
    """Fetch closed issues + PRs from the GitHub API."""
    # We fetch both issues and PRs in one call (GitHub's issues endpoint returns both)
    url = f"{_GITHUB_API}/repos/{repo}/issues?state=closed&per_page={min(limit, 100)}"
    data = _get(url)
    return data if isinstance(data, list) else []


def _process_items(raw_items: list[dict]) -> tuple[list[Bug], dict]:
    """Extract Bug objects and developer stats from raw GitHub API responses."""
    bugs: list[Bug] = []
    dev_stats: dict = {}

    for item in raw_items:
        # Determine developer: prefer formal assignee, fall back to PR author
        dev_username = _extract_developer(item)
        if not dev_username:
            continue

        bug_id      = item["number"]
        title       = item.get("title", "")
        body        = item.get("body") or ""
        description = f"{title} {body}".strip()

        labels    = [lbl["name"].lower() for lbl in item.get("labels", [])]
        severity  = _severity_from_labels(labels)
        component = _component_from_labels(labels)
        fix_time  = _parse_fix_time(item.get("created_at"), item.get("closed_at"))

        bug = Bug(id=bug_id, description=description, severity=severity, module=component)
        bugs.append(bug)

        if dev_username not in dev_stats:
            dev_stats[dev_username] = {
                "name": dev_username,
                "bugs_fixed": 0,
                "total_fix_time": 0.0,
                "components": set(),
                "resolved_bug_ids": [],
            }

        dev_stats[dev_username]["bugs_fixed"]      += 1
        dev_stats[dev_username]["total_fix_time"]  += fix_time
        dev_stats[dev_username]["components"].add(component)
        dev_stats[dev_username]["resolved_bug_ids"].append(bug_id)

    return bugs, dev_stats


def _extract_developer(item: dict) -> str | None:
    """
    Return the GitHub username of the developer responsible for closing this item.

    Priority:
      1. First formal assignee (used by vscode, angular, golang, etc.)
      2. PR author            (used by react, chromium, etc.)
    """
    assignees = item.get("assignees", [])
    if assignees:
        return assignees[0]["login"]

    # Fall back to the author if this is a PR
    if "pull_request" in item:
        user = item.get("user")
        if user:
            return user["login"]

    return None


def _severity_from_labels(labels: list[str]) -> str:
    for lbl in labels:
        sev = map_severity(lbl)
        if sev != "normal":
            return sev
    return "normal"


def _component_from_labels(labels: list[str]) -> str:
    for lbl in labels:
        if lbl.startswith(("area-", "component-", "pkg/", "team-")):
            return lbl
    return "general"


def _parse_fix_time(
    created_at: str | None,
    closed_at: str | None,
    fallback_hours: float = 24.0,
) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        t0 = datetime.datetime.strptime(created_at, fmt)
        t1 = datetime.datetime.strptime(closed_at, fmt)
        return max(0.1, (t1 - t0).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return fallback_hours


def _get(url: str) -> dict | list | None:
    """Make a GET request to the GitHub API and return parsed JSON."""
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[GitHub Loader] HTTP {e.code} fetching {url}")
        return None
    except Exception as e:
        print(f"[GitHub Loader] Error fetching {url}: {e}")
        return None
