#!/bin/bash
# Daily auto-commit: commits any pending changes in the repo and pushes.
# Cycles through MESSAGES in order, picking the one after the previous
# commit's message. Does nothing if there are no changes. Scheduled by launchd
# (see install.sh). Safe to run by hand too.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

MESSAGES=(
  "small tweaks"
  "minor updates"
  "cleaning things up"
  "quick fixes"
  "a bit of polish"
)

git add -A   # .gitignore keeps .env, .venv and caches out
if git diff --cached --quiet; then
  echo "$(date): nothing to commit"
  exit 0
fi

# Next message in the cycle after the last one used (or the first, if the
# last commit wasn't an auto-commit).
LAST="$(git log -1 --pretty=%s 2>/dev/null)"
NEXT=0
for i in "${!MESSAGES[@]}"; do
  if [ "${MESSAGES[$i]}" = "$LAST" ]; then
    NEXT=$(( (i + 1) % ${#MESSAGES[@]} ))
  fi
done
MSG="${MESSAGES[$NEXT]}"

git commit -q -m "$MSG" || exit 1
git push -q origin HEAD 2>&1 || echo "$(date): push failed (offline?) - will go out with the next run"
echo "$(date): committed \"$MSG\""
