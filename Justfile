
build:
    #!/usr/bin/env bash
    # need shebang for if statements; see:
    # https://just.systems/man/en/multi-line-constructs.html

    # check worktree subfolder exists
    if ! [[ -d .wt/gh-pages/ ]]; then
        echo "No '.wt/gh-pages' folder found"
        echo "Run: 'mkdir -p .wt && git worktree add .wt/gh-pages gh-pages'" \
                "then try again"
        exit 1
    fi
    # check theme submodule has been cloned correctly
    if [[ -z $(ls -A src/themes/beautifulhugo) ]]; then
        echo "The themes subfolder is empty"
        echo "Run: 'git submodule update --init --recursive' then try again"
        exit 1
    fi
    # build and sync website
    (cd src && hugo)
    rsync -a docs/ .wt/gh-pages/docs/
    touch .wt/gh-pages/docs/.nojekyll

dev:
    (cd src && just dev)
