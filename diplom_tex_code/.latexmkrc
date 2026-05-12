# latexmk config for this project
# Forces XeLaTeX + biber, runs the right number of passes for bib + cleveref.

$pdf_mode = 5;            # 5 = xelatex
$bibtex_use = 2;          # always run bib
$biber = 'biber %O %B';   # use biber (we're on biblatex)
$max_repeat = 5;

# Keep aux files in the same folder; clean targets
$clean_ext = "synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk fls log aux out toc";
