using ARGridBootstrap
using Test, Random

# Create fake RNG that generates 1:T to pass to simulate_estimate_arp
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

@testset "cycle" begin
  T = 25
  foo = cycle(T)
  for t in 1:(T+5)
    @test foo()==((t-1)%T + 1)
  end
  @test foo(3)==6:8
end

@testset "AR estimation and simulation" begin

  T = 200
  e = randn(T)
  y0 = 0
  α = 0.9

  y = ar1_original(y0, α, e, cycle(T))
  @test length(y)==T
  (bo, so, eo) = b_est_original(y)
  (bm, sm, em) = b_est_mldivide(y)
  @test bo ≈ bm
  @test so ≈ sm
  @test eo ≈ em
  (bx, sx, ex) = b_est_nox(y)
  @test bo ≈ bx
  @test so ≈ sx
  @test eo ≈ ex

  (bs, ss) = simulate_estimate_arp(y0, α, e, Val(1),cycle(T))
  @test bo ≈ bs
  @test so ≈ ss
end

@testset "Gridbootstrap" begin

  T = 200
  e = randn(T)
  y0 = 0
  α = 0.9
  grid = 0.5:(0.56/20):1.06  
  seed=78453 
  Random.seed!(seed)
  estimator(x)=begin
    out=b_est_original(x)
    (out.θ[3], out.se[3])
  end
  (bo, to) = gridbootstrap(estimator, a->ar1_original(y0, a, e),grid , 19)  
  Random.seed!(seed)
  estimator(x)=begin
    out=b_est_nox(x)
    (out.θ[3], out.se[3])
  end
  (bx, tx) = gridbootstrap(estimator, a->ar1_original(y0, a, e), grid, 19)
  @test bo ≈ bx
  @test isapprox(to, tx, rtol=1e-4) 
  Random.seed!(seed)
  estimator(a)=begin
    out=simulate_estimate_arp(y0,a,e)
    (out.θ[3], out.se[3])
  end
  (bs, ts) = gridbootstrap(estimator, a->a, grid, 19)  
  @test bx ≈ bs
  @test isapprox(tx, ts, rtol=1e-4)    
end

@testset "Gridbootstrap_threaded" begin
  T = 200
  e = randn(T)
  y0 = 0
  α = 0.9
  grid = 0.5:(0.56/20):1.06  
  rng  = rngarray(Base.Threads.nthreads())
  estimator(x)=begin
    out=b_est_nox(x)
    (out.θ[3], out.se[3])
  end
  (bx, tx) = gridbootstrap_threaded(estimator,
                                    (a, rng)->ar1_original(y0, a, e, n->rand(rng,1:(T-1),n)),
                                    grid, 19, rng=deepcopy(rng))  
  estimator(foo)=begin
    (a, rng) = foo
    out=simulate_estimate_arp(y0,a,e,Val(1),n->rand(rng,1:(T-1),n))
    (out.θ[3], out.se[3])
  end
  (bs, ts) = gridbootstrap_threaded(estimator, (a,rng)->(a,rng), grid,
                                    19, rng=deepcopy(rng))  
  @test bx ≈ bs
  @test isapprox(tx, ts, rtol=1e-4)  
end


@testset "gridbootstrap_gpu" begin
  
  T = 200
  e = randn(T)
  y0 = 0
  grid = 0.5:(0.56/20):1.06
  nboot=199

  (b,t) = argridbootstrap_gpu(e,y0,grid=grid, nboot=nboot)
  @test size(b) == (nboot, length(grid))
  @test size(t) == (nboot, length(grid))
  
end
