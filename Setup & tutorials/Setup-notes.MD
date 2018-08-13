## Installations:
- ATOM packages: script, pdf-view, language-latex, hydrogen-launcher, file-icons, atom-latex
- TeXLive works with atom-latex, MiKTeX does not!
- Python is installed by installing Anaconda (choose "Add Anaconda to my PATH environment variable"), and running it through hydrogen in atom.
    i kommandprompt skriv: "apm" (tjek om det findes) og "apm install hydrogen"
    add Script and optional packages from https://www.pythonmania.net/en/2017/02/27/recommended-atom-packages/
- R is installed from the R website, and Rstudio from the Rstudio website (If you install IRkernel (google it) you can run R in atom through hydrogen)

## Spell-check:
Pre-installed Spell-check package
- Add text.tex.latex to "Grammars"
- Add the following line to "Excluded Scopes":
    support.function.tex, support.function.textit.latex, support.function.textbf.latex
- Add "en-US, da-DK, es-ES" or others under locales (check "Use Locales").
  Check that English (United States) etc. is installed in Windows language settings.
