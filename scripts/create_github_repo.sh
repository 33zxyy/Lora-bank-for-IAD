#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GITHUB_TOKEN=xxx GITHUB_OWNER=your_name ./scripts/create_github_repo.sh
# Optional env:
#   REPO_NAME (default: original-cdad)
#   VISIBILITY (public|private, default: public)

REPO_NAME="${REPO_NAME:-original-cdad}"
VISIBILITY="${VISIBILITY:-public}"
GITHUB_OWNER="${GITHUB_OWNER:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ -z "${GITHUB_OWNER}" ]]; then
  echo "[ERROR] Missing GITHUB_OWNER env (GitHub user or org)." >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN}" ]]; then
  echo "[ERROR] Missing GITHUB_TOKEN env (needs repo scope)." >&2
  exit 1
fi

if [[ "${VISIBILITY}" != "public" && "${VISIBILITY}" != "private" ]]; then
  echo "[ERROR] VISIBILITY must be 'public' or 'private'." >&2
  exit 1
fi

# Build payload without jq dependency.
if [[ "${VISIBILITY}" == "private" ]]; then
  PRIVATE_JSON=true
else
  PRIVATE_JSON=false
fi

CREATE_PAYLOAD=$(cat <<JSON
{"name":"${REPO_NAME}","private":${PRIVATE_JSON},"auto_init":false}
JSON
)

# Try user repo creation first.
CREATE_URL="https://api.github.com/user/repos"
HTTP_CODE=$(curl -sS -o /tmp/create_repo_resp.json -w "%{http_code}" \
  -X POST "${CREATE_URL}" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "${CREATE_PAYLOAD}")

if [[ "${HTTP_CODE}" != "201" ]]; then
  # If user creation failed, try org route for cases where owner is an org.
  ORG_URL="https://api.github.com/orgs/${GITHUB_OWNER}/repos"
  HTTP_CODE=$(curl -sS -o /tmp/create_repo_resp.json -w "%{http_code}" \
    -X POST "${ORG_URL}" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "${CREATE_PAYLOAD}")
fi

if [[ "${HTTP_CODE}" != "201" && "${HTTP_CODE}" != "422" ]]; then
  echo "[ERROR] GitHub API failed with HTTP ${HTTP_CODE}." >&2
  cat /tmp/create_repo_resp.json >&2 || true
  exit 1
fi

TARGET_URL="https://github.com/${GITHUB_OWNER}/${REPO_NAME}.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${TARGET_URL}"
else
  git remote add origin "${TARGET_URL}"
fi

# Push the current branch to new repository.
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push -u "https://${GITHUB_OWNER}:${GITHUB_TOKEN}@github.com/${GITHUB_OWNER}/${REPO_NAME}.git" "${CURRENT_BRANCH}"

echo "[OK] Repository is ready: https://github.com/${GITHUB_OWNER}/${REPO_NAME}"
