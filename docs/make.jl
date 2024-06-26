using Documenter, DocumenterMarkdown, ARGridBootstrap

runweave=true
runnotebook=false

if runweave
  using Weave
  wd = pwd()
  try
    builddir=joinpath(dirname(Base.pathof(ARGridBootstrap)),"..","docs","build")
    mkpath(builddir)
    cd(builddir)
    jmdfiles = filter(x->occursin(r".jmd$",x), readdir(joinpath("..","jmd")))
    for f in jmdfiles
      src = joinpath("..","jmd",f)
      target = joinpath("..","build",replace(f, r"jmd$"=>s"md"))
      if stat(src).mtime > stat(target).mtime
        weave(src,out_path=joinpath("..","build"),
              cache=:refresh,
              cache_path=joinpath("..","weavecache"),
              doctype="github", mod=Main,
              args=Dict("md" => true))
      end
      target = joinpath("..","build",replace(f, r"jmd$"=>s"ipynb"))
      if (runnotebook && stat(src).mtime > stat(target).mtime)
          notebook(src,out_path=joinpath("..","build"),
                   nbconvert_options="--allow-errors")
      elseif (stat(src).mtime > stat(target).mtime)
        convert_doc(src, joinpath("..","build",replace(f, "jmd" => "ipynb")))
      end
    end
  finally
    cd(wd)
  end
  if (isfile("build/temp.md"))
    rm("build/temp.md")
  end
end

makedocs(
  modules=[ARGridBootstrap],
  format=Markdown(),
  clean=false,
  pages=[
    "Home" => "index.md", # this won't get used anyway; we use quarto instead for interoperability with weave's markdown output.
  ],
  repo="https://github.com/schrimpf/ARGridBootstrap.jl/blob/{commit}{path}#L{line}",
  sitename="ARGridBootstrap.jl",
  authors="Paul Schrimpf <paul.schrimpf@gmail.com>",
)

#run(`quarto build build`)

deploy=false
if deploy || "deploy" in ARGS
  cd(@__DIR__)
  run(`quarto publish gh-pages build`)
end
