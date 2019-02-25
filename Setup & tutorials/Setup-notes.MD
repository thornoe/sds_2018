## General installation of Atom:
-   <https://flight-manual.atom.io/getting-started/sections/installing-atom/>
-   Always install the Atom packages: script, hydrogen-launcher, file-icons, pdf-view, atom-latex, language-latex

## LaTex
-   TeXLive works with atom-latex, MiKTeX does not!
-   For optional packages see <https://gist.github.com/Aerijo/5b9522530715e5be6e89fc012e9a72a8> (ignore the Linter-packages)

## Python
-   Python is installed by installing Anaconda (choose "Add Anaconda to my PATH environment variable"), and running it through hydrogen in atom.
    - In Command Prompt run the command `apm` (checking apm is installed) and `apm install hydrogen`
    - Add Script and optional packages from this list <https://www.pythonmania.net/en/2017/02/27/recommended-atom-packages/>
    - Install the Python language server (pyls) by running `pip3 install --user python-language-server`
    - Update Python once in a while `conda update --all`

## R language
-   R is installed from the R website, and Rstudio from the Rstudio website (If you install IRkernel (google it) you can run R in atom through hydrogen)
-   Install according to <https://jstaf.github.io/2018/03/25/atom-ide.html/>
-   OBS: The atom-ide-ui terminal replaces Linter (i.e. ignore the Linter part below)

## Spell-check:
Pre-installed Spell-check package
-   Add `text.tex.latex, text.tex.latex.beamer` to "Grammars"
    - To find out what scope to add for a current file: put the cursor in the file, open the Command Palette `ctrl-shift-p`, and search for the `Editor: Log Cursor Scope` command.
-   Add the following line to "Excluded Scopes":
    - `Support.function.tex, support.function.latex, meta.preamble.latex, support.type.function.latex, comment.line.percentage.tex, storage.type.function.latex, string.other.math.tex, string.other.math.block.environment.latex, variable.parameter.function.latex, constant.other.reference.latex`
-   Add `en-US, da-DK, es-ES` or others under locales (check "Use Locales").
-   Windows Language (equivalently for Linux):
    - Check that `English (United States)` is installed in Windows language settings.
    - For now you need to set the system language to `English (United States)` as well for spell-check to work.

## Ide
Used for language support instead of Linter. Run `apm install ...` in Command Prompt
-   atom-ide-ui
-   ide-python
-   ide-r
-   platformio-ide-terminal (optional terminal package for atom)

## Linter
Common package for language support but suboptimal for Python & R
-   linter-pydocstyle & flake8 for Python (Ignore Error Codes: D100, E266, E305, W391 in flake8)
-   linter-ui-default & linter-chktex for LaTex
-   linter-Markdown
