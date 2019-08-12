## General installation of Atom:
-   Follow the installation guide <https://flight-manual.atom.io/getting-started/sections/installing-atom/>
-   To install Atom packages either
    - Press `ctrl+,` in Atom and search for packages under 'install'
    - Open the terminal (e.g. Command Prompt in Windows) and run a command like `apm install script`. The first time run `apm` to check apm is installed correctly.
-   Always install the following Atom packages by running the command:
    ```
    apm install script hydrogen-launcher file-icons pdf-view atom-latex language-latex 
    ```

## LaTeX
-   TeXLive works with atom-latex, MiKTeX does not!
-   Check the [optional packages listed by Benjamin Gray](https://gist.github.com/Aerijo/5b9522530715e5be6e89fc012e9a72a8) (ignore the Linter-packages and see 'Ide' below instead)

## Markdown
-   Install `language-markdown` and `markdown-writer`
-   `markdown-preview` is pre-installed. Press `ctrl+shift+m` to toogle on/off.

## Python
-   Python is installed by installing [Anaconda](https://www.anaconda.com/distribution/#download-section) (choose 'Add Anaconda to my PATH environment variable'), and running it through hydrogen in atom.
-   In the terminal run `apm install hydrogen`
-   All you need is Script, but consider optional packages from the [list at pythonmania](https://www.pythonmania.net/en/2017/02/27/recommended-atom-packages) (ignore `linter-pydocstyle` & `flake8` but see IDE below instead).
-   Install the Python language server (pyls) by running `pip3 install --user python-language-server` in the terminal
-   Update Python once in a while `conda update --all` (and the [hack](https://pypi.org/project/pip-upgrade-outdated/) `pip_upgrade_outdated -3`).

## R
-   R is installed from the R website, and Rstudio from the Rstudio website (If you install IRkernel (google it) you can run R in atom through hydrogen)
-   Install according to the [guide by Jeff Stafford](https://jstaf.github.io/2018/03/25/atom-ide.html)
-   OBS: The atom-ide-ui terminal replaces Linter (i.e. see the Ide not the Linter part below)

## Spell-check
Pre-installed Spell-check package
-   Add `text.tex.latex, text.tex.latex.beamer, text.md` to "Grammars"
    - To find out what scope to add for a current file: put the cursor in the file, open the Command Palette `ctrl-shift-p`, and search for the `Editor: Log Cursor Scope` command.
-   Add the following line to "Excluded Scopes":
    - `markup.underline.link.gfm, Support.function.tex, support.function.latex, meta.preamble.latex, support.type.function.latex, comment.line.percentage.tex, storage.type.function.latex, string.other.math.tex, string.other.math.block.environment.latex, variable.parameter.function.latex, constant.other.reference.latex`
-   Add `en-US`, `da-DK`, `es-ES` or others under 'Locales' (check the 'Use Locales' box).
-   Windows Language (equivalently for Linux):
    - For now you need to set the system language to `English (United States)` for spell-check to work.
    - When a certain bug fix is approved you'll only need to make sure that `English (United States)` is installed in Windows language settings.
-   Place the cursor at a highlighted word and press `ctrl+shift+:` to run spell-check

## IDE
Used for language support instead of Linter due to better functionality with Python and R. Install the following Atom packages
-   `atom-ide-ui` (Under 'Settings' > 'Enabled Features' I choose to disable 'Console', 'Debugger', 'atom-ide-refactor', and 'Terminal')
-   `ide-python`
    - I add the [error codes](http://pycodestyle.pycqa.org/en/latest/intro.html#error-codes) `E2, E3, E501, W2, W3, W5` under 'Settings' > 'PyCodeStyle' > 'Ignore'
    - Automatically changes tab-length to 4 spaces for all .py files (overwrites global Atom settings)
-   `ide-r`
-   `atom-ide-debugger-python`(optional debugger)
-   `platformio-ide-terminal` (optional terminal package for atom)

## Linter
Common package for language support but suboptimal for Python and R, thus, I recommend using IDE instead.
-   However, if you prefer Linter (e.g. if you only use Atom for LaTeX and Markdown) you can install the package `linter` and the relevant [language specific linter packages](https://atomlinter.github.io/), e.g.
    - `linter-markdown`
    - `linter-ui-default` & `linter-chktex` for LaTeX
    - `linter-pydocstyle` & `flake8` for Python (Ignore Error Codes: D100, E266, E305, W391 in flake8)
