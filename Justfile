build:
    (cd src && hugo)
    rm -rf .wt/gh-pages/docs
    rsync -a --delete docs/ .wt/gh-pages/docs/
    touch docs/.nojekyll
    [[ -f .wt/gh-pages/docs/.nojekyll ]] || touch .wt/gh-pages/docs/.nojekyll
