# ake RNG that generates 1:T to pass to simulate_estimate_arp
function cycle(T)
    state = 0
    function next(n=1)
      if (n==1) 
        state = (state>=T) ? 1 : state+1
        return(state)
      else
        out = zeros(Int,n)
        for i in eachindex(out)
          out[i] = next()
        end
        return(out)
      end
    end
  end
  