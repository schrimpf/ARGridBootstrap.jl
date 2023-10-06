function functiontext(functionname, filename; includedoc=true)
  lines = readlines(filename)
  fstart=findfirst(occursin.(Regex("function\\s+$(functionname)"),lines))
  fend  =fstart + findfirst(occursin.(r"^end",lines[(fstart+1):end]))
  
  if (includedoc && occursin(r"^\"\"\"",lines[fstart-1]) )
    dend = fstart -1
    dstart = dend - findfirst(occursin.(r"^\"\"\"", lines[(fstart-2):(-1):1]))
    println("doc lines $dstart - $dend")                  
  end
  println("function lines $fstart - $fend")
  lines[dstart:fend]
end
