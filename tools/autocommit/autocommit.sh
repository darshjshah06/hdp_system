#!/bin/bash
# Daily auto-commit: commits any pending changes in the repo (or an empty
# commit if there are none) and pushes. The message is picked at random from
# MESSAGES, never repeating the previous commit's. Scheduled by launchd (see
# install.sh). Safe to run by hand too.
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
# Random message, skipping whichever one the last commit used.
LAST="$(git log -1 --pretty=%s 2>/dev/null)"
MSG="$LAST"
while [ "$MSG" = "$LAST" ]; do
  MSG="${MESSAGES[RANDOM % ${#MESSAGES[@]}]}"
done

git commit -q --allow-empty -m "$MSG" || exit 1
git push -q origin HEAD 2>&1 || echo "$(date): push failed (offline?) - will go out with the next run"
echo "$(date): committed \"$MSG\""
