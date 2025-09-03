build:
    (cd src && hugo)
    [[ -d .wt/gh-pages/ ]] || echo "Create first a git worktree '.wt/gh-pages'"
    rsync -a --delete docs/ .wt/gh-pages/docs/
    touch .wt/gh-pages/docs/.nojekyll
