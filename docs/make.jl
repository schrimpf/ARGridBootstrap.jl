using Documenter, Weave, ARGridBootstrap

weave("jmd/argridboot.jmd", cache=:user,
      doctype="pandoc2html", out_path=joinpath(@__DIR__,"src"),
      pandoc_options=["--toc","--toc-depth=2","--filter=pandoc-citeproc"])

makedocs(sitename="ARGridBootstrap",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
         pages = ["index.md",
                  "test.md"]
         )
