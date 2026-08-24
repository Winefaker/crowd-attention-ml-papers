#!/bin/bash
# Publish this project to GitHub.
#
# Run this ONCE, after `gh auth login`:
#     bash publish_dashboard.sh
#
# It creates a PRIVATE repository from this folder and pushes it. Nothing is visible
# to anyone else. Re-running it just pushes the current contents again.
#
# To publish publicly instead (this also turns on GitHub Pages, which serves the
# dashboard at a link you can share):
#     VISIBILITY=public bash publish_dashboard.sh
#
# Repository name. Override it on the command line if you prefer another:
#     REPO=my-repo-name bash publish_dashboard.sh
REPO=${REPO:-crowd-attention-ml-papers}
VISIBILITY=${VISIBILITY:-private}

set -euo pipefail
cd "$(dirname "$0")"

case "$VISIBILITY" in
  private|public) ;;
  *) echo "VISIBILITY must be private or public, got '$VISIBILITY'"; exit 1 ;;
esac

command -v gh >/dev/null || { echo "GitHub CLI not found. Install it with: brew install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run this first, then re-run me:"; echo "    gh auth login"; exit 1; }

# Let git itself authenticate to github.com over https.
gh auth setup-git >/dev/null 2>&1 || true

LOGIN=$(gh api user --jq .login)
UID_GH=$(gh api user --jq .id)
URL="https://$(echo "$LOGIN" | tr "[:upper:]" "[:lower:]").github.io/$REPO/"
echo "Publishing as $LOGIN to $REPO ($VISIBILITY)"

# Commit identity: the GitHub noreply address, so the personal email stays private.
# Set on every run, not only when this script creates the repository.
[ -d .git ] || git init -q -b main
git config user.name "$LOGIN"
git config user.email "${UID_GH}+${LOGIN}@users.noreply.github.com"

git add -A
git diff --cached --quiet || git commit -q -m "Community attention and citation impact of ML papers: analysis code and dashboard"

if gh repo view "$LOGIN/$REPO" >/dev/null 2>&1; then
  echo "Repository $LOGIN/$REPO already exists, pushing to it."
  git remote set-url origin "https://github.com/$LOGIN/$REPO.git" 2>/dev/null \
    || git remote add origin "https://github.com/$LOGIN/$REPO.git"
  git push -u origin main
else
  gh repo create "$REPO" --"$VISIBILITY" --source=. --remote=origin --push \
    --description "Does community attention predict the citation impact of ML papers? Analysis code and an interactive dashboard."
fi

# ---------------------------------------------------------------- GitHub Pages ---
# Pages is deliberately NOT touched while the repository is private. Two reasons:
# on a free account Pages needs a public repository anyway, and on a paid account a
# Pages site is readable by anyone on the internet even when its repository is
# private. Turning it on here would publish the dashboard against your intent.
PAGES_ON=no
if [ "$VISIBILITY" = "public" ]; then
  if ERR=$(gh api -X POST "repos/$LOGIN/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" 2>&1); then
    PAGES_ON=yes
    URL=$(gh api "repos/$LOGIN/$REPO/pages" --jq .html_url 2>/dev/null || echo "$URL")
    echo "GitHub Pages enabled."
  elif ERR2=$(gh api -X PUT "repos/$LOGIN/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" 2>&1); then
    PAGES_ON=yes
    URL=$(gh api "repos/$LOGIN/$REPO/pages" --jq .html_url 2>/dev/null || echo "$URL")
    echo "GitHub Pages already on, source confirmed."
  else
    echo "Could not turn on Pages automatically:"
    echo "  $ERR" | head -3
    echo "Turn it on by hand: https://github.com/$LOGIN/$REPO/settings/pages (Source: deploy from branch, main, / root)"
  fi
else
  echo "Pages left off: a Pages site is readable by anyone on the internet, even when its repository is private."
fi

# ------------------------------------------------------- the README's link line --
# Point the README at the hosted dashboard if it exists, at the local file if it does not.
# Runs every time, so a repository that goes from private to public gets its link back.
python3 - "$URL" "$PAGES_ON" <<'PY'
import sys, pathlib, re
url, pages_on = sys.argv[1], sys.argv[2] == "yes"
p = pathlib.Path("README.md")
t = p.read_text()

hosted = ("**[Open the interactive dashboard](" + url + ")**\n\n"
          "If that link is not live yet, clone the repository and open `index.html` in any browser. It needs no\n"
          "server and no network.")
local = ("**Open the dashboard:** clone this repository and open `index.html` in any browser. It needs no\n"
         "server and no network.\n\n"
         "If this repository is made public, the hosted version appears at " + url)
want = hosted if pages_on else local

# Match the link block whichever form it is in, ignoring the URL and its casing.
pattern = re.compile(
    r"(?:\*\*\[Open the interactive dashboard\]\([^)]*\)\*\*\s*\n\n"
    r"If that link is not live yet.*?server and no network\.)"
    r"|"
    r"(?:\*\*Open the dashboard:\*\*.*?server and no network\."
    r"(?:\s*\n\nIf this repository is made public, the hosted version appears at \S+)?)",
    re.S)
t2, n = pattern.subn(lambda _: want, t, count=1)
if n == 0:                      # nothing recognisable: fall back to the placeholder
    t2 = t.replace("DASHBOARD_URL", url)
p.write_text(t2)
PY
git add -A
git diff --cached --quiet || { git commit -q -m "Point the README at the dashboard"; git push -q origin main; }

echo
echo "Repository: https://github.com/$LOGIN/$REPO  ($VISIBILITY)"

if [ "$PAGES_ON" != "yes" ]; then
  cat <<EOF

The dashboard is not hosted, because the repository is private.
To view it now:      open index.html in this folder.
To share it later, make the repository public and re-run this script:

    gh repo edit $LOGIN/$REPO --visibility public --accept-visibility-change-consequences
    VISIBILITY=public bash publish_dashboard.sh

The second command turns Pages on and prints the live link.
EOF
  exit 0
fi

echo "Dashboard:  $URL"
echo
echo "The first Pages build takes a minute or two. Checking..."
CODE=000
for i in $(seq 1 20); do
  CODE=$(curl -sL -o /dev/null -w '%{http_code}' "$URL") || CODE=000
  if [ "$CODE" = "200" ]; then
    echo "Live: $URL"
    exit 0
  fi
  if [ "$i" -lt 20 ]; then sleep 15; fi
done
echo "Not live yet (last status $CODE). It usually appears within a few minutes at $URL"
echo "You can watch the build at https://github.com/$LOGIN/$REPO/actions"
