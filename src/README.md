# Editing the course website

## Cloning the website

```bash
git clone --recurse-submodules git@github.com:ai-for-humanity-ucph/2025.git
```

then

```bash
cd 2025/src
```

and follow [Hugo](#hugo).

Most of the website is constructed from the markdown files are inside
[src/content](content/).
E.g. the schedule is [this](content/schedule/index.md) file constructed with
data from the `yaml` file [here](data/schedule.yaml).

## Hugo

- Download hugo from [here](https://gohugo.io/installation/)
- Check that the executable works in your terminal by typing

```bash
❯ hugo version
hugo v0.147.9+extended+withdeploy linux/amd64 BuildDate=unknown
```

### Live editing

- From inside the `src` folder type `hugo server` with some extra flags to
  avoid cache issues; you should get the following output:

```bash
❯ hugo server --ignoreCache --disableFastRender --cleanDestinationDir
Watching for changes in /home/jsr-p/Dropbox/sodas/project/ai-4-humanity/repo-2025/src/{content,layouts,static,themes}
Watching for config changes in /home/jsr-p/Dropbox/sodas/project/ai-4-humanity/repo-2025/src/config.toml
Start building sites …
hugo v0.147.9+extended+withdeploy linux/amd64 BuildDate=unknown

WARN  calling IsSet with unsupported type "ptr" (*page.siteWrapper) will always return false.
WARN  calling IsSet with unsupported type "invalid" (<nil>) will always return false.

                  │ EN
──────────────────┼─────
 Pages            │  12
 Paginator pages  │   0
 Non-page files   │   1
 Static files     │ 185
 Processed images │   0
 Aliases          │   3
 Cleaned          │   0

Built in 26 ms
Environment: "development"
Serving pages from disk
Running in Fast Render Mode. For full rebuilds on change: hugo server --disableFastRender
Web Server is available at http://localhost:1313/2025/ (bind address 127.0.0.1)
Press Ctrl+C to stop
```

- You can now edit the markdown files and see the changes instantly in the
  browser

### Building the site

- From inside the `src` folder type `hugo`

```bash
❯ hugo
Start building sites …
hugo v0.147.9+extended+withdeploy linux/amd64 BuildDate=unknown

WARN  calling IsSet with unsupported type "ptr" (*page.siteWrapper) will always return false.
WARN  calling IsSet with unsupported type "invalid" (<nil>) will always return false.

                  │ EN
──────────────────┼─────
 Pages            │  12
 Paginator pages  │   0
 Non-page files   │   1
 Static files     │ 185
 Processed images │   0
 Aliases          │   3
 Cleaned          │   0

Total in 50 ms
```

- The website will be built to the `../docs` folder
- We will later deploy the website from the `gh-pages` branch
