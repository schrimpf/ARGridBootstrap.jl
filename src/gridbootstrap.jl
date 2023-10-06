"""
    gridbootstrap(estimator, simulator,
                  grid::AbstractVector,
                  nboot=199)

Computes grid bootstrap estimates a single parameter model.

For each α ∈ grid, repeatedly simulate data with parameter α and then compute an estimate.


# Arguments
- `estimator` function of output of `simulator` that returns a
        2-tuple containing an estimate of α and its standard error.
- `simulator` function that given `α` simulates data that can be used to estimate α
- `grid` grid of parameter values. For each value, `nboot`
        datasets will be simulated and estimates computed.
- `nboot`

# Returns
- `ba` hatα - α for each grid value and simulated dataset
- `t` t-stat  for each grid value and simulated dataset
"""
function gridbootstrap(estimator, simulator,
                       grid,
                       nboot=199)
  g = length(grid)
  bootq = zeros(nboot, g)
  ba    = zeros(nboot, g)
  bootse = zeros(nboot,g)
  for ak in 1:g
    for j in 1:nboot
      (bootq[j,ak], bootse[j,ak]) = estimator(simulator(grid[ak]))
      ba[j,ak] = bootq[j,ak] - grid[ak]
    end
  end
  ts = ba./bootse
  (ba=ba, t=ts)
end


"""
    gridbootstrap_threaded(estimator, simulator,
                    grid::AbstractVector,
                    nboot=199, rng=rngarray(nthreads())

Computes grid bootstrap estimates a single parameter model.

Multithreaded version.

For each α ∈ grid, repeatedly simulate data with parameter α and then compute an estimate.


# Arguments
- `estimator` function of output of `simulator` that returns a
    2-tuple containing an estimate of α and its standard error.
- `simulator` function that given `α` and `rng`, simulates data
    that can be used to estimate α
- `grid` grid of parameter values. For each value, `nboot`
    datasets will be simulated and estimates computed.
- `nboot` number of bootstrap simulations per grid point

# Returns
- `ba` hatα - α for each grid value and simulated dataset
- `t` t-stat  for each grid value and simulated dataset
"""
function gridbootstrap_threaded(estimator, simulator,
                                grid::AbstractVector,
                                nboot=199)
  g = length(grid)
  bootq = zeros(nboot, g)
  ba    = zeros(nboot, g)
  bootse = zeros(nboot,g)
  #@threads for ak in 1:g
  #  for j in 1:nboot
  @threads for ind ∈ CartesianIndices(ba)
    j = ind[1]
    ak = ind[2]
    (bootq[j,ak], bootse[j,ak]) = estimator(simulator(grid[ak],Random.TaskLocalRNG()))
    ba[j,ak] = bootq[j,ak] - grid[ak]
  end
  ts = ba./bootse
  (ba=ba, t=ts)
end


"""
    rngarray(n)

  Create `n` rng states that will not overlap for 10^20 steps.

  Note: this will be unneeded in Julia 1.3 when thread-safe RNG is
  included.
"""
function rngarray(n)
  baserng =  MersenneTwister()
  rng = Array{typeof(baserng)}(undef, Base.Threads.nthreads())
  rng[1] = baserng
  steps = big(10)^20 # randjump is precomputed for steps = big(10)^20
  for i in 2:nthreads()
    rng[i] = Future.randjump(rng[i-1], steps)
  end
  rng
end


################################################################################
"""
    argridbootstrap_gpu(e; αgrid = 0.84:(0.22/20):1.06,
                          nboot=199, RealType = Float32)

Computes grid bootstrap estimates for an AR(1) model.

For each α ∈ grid, repeatedly simulate data with parameter α and then compute an estimate.

# Arguments
- `e` vector error terms that will be resampled with replacement
        to generate bootstrap sample
- `grid` grid of parameter values. For each value, `nboot`
        datasets will be simulated and estimates computed.
- `nboot`
- `RealType` type of numbers for GPU computation. On many GPUs,
        Float32 will have better performance than Float64.

# Returns
- `ba` hatα - α for each grid value and simulated dataset
- `t` t-stat  for each grid value and simulated dataset
"""
function argridbootstrap_gpu(e, y0;
                             grid = 0.84:(0.22/20):1.06,
                             nboot=199, RealType = Float32)
  g = length(grid)

  P = 3
  # Allocate GPU memory
  bootq = CuArray(zeros(RealType, nboot, g))
  ba    = CuArray(zeros(RealType, nboot, g))
  bootse= CuArray(zeros(RealType, nboot,g))
  αg = CuArray(RealType.(grid))
  eg = CuArray(RealType.(e))
  ei = Int.(ceil.(length(e).*CUDA.rand(RealType,nboot,g,length(e))))

  # use of registers in gridkernel! limits the maximum threads to less
  # than the full 1024
  maxthreads = sizeof(RealType)<=4 ? 512 : 256
  gthreads =2^2
  bthreads =maxthreads ÷ gthreads
  bblocks = Int(ceil(nboot/bthreads))
  gblocks = Int(ceil(g/gthreads))

  @cuda threads=bthreads,gthreads blocks=bblocks,gblocks argridkernel!(ba,bootq,bootse,Val(1), eg, ei, αg)
  ts = ba./bootse
  (ba=collect(ba), t=collect(ts))
end

"""
    argridkernel!(ba,bootq, bootse, ar::Val{P}, e, ei, αgrid)

GPU kernel for simulation and estimation of AR(P) model.

# Arguments (modified on return)
- `ba`: `nboot × ngrid` array.  Will be filled with bootstrap estimates of α
   grid values of true α
- `bootq`: `nboot × ngrid` array.  Will be filled with bootstrap
   estimates of α
- `bootse`: `nboot × ngrid` array.  Will be filled with standard
   errors of α for each bootstrap repetition


# Arguments (not modified)
- `ar::Val{P}` : autoregressive order for estimation. Simulated
   model will always be AR(1) with 0 intercept and time trend, but
   estimation will use an AR(P) model with intercept and time
   trend. Only the AR(1) parameter estimate is included in `ba`,
   `bootq`, and `bootse`.
- `e` : error terms to draw with replacement
- `ei` : `nboot × ngrid × length(e)` array of indices of `e` to
         use to generate bootstrap sample1
- `αgrid` : length `ngrid` values of AR(1) parameter to perform
   bootstrap on.

Returns nothing, but modifies in place `ba`, `bootq`, and `bootse`
"""
function argridkernel!(ba,bootq, bootse,
                       ar::Val{P}, e, ei, αgrid) where P
  b = threadIdx().x +  (blockIdx().x-1)*blockDim().x
  ak= threadIdx().y + (blockIdx().y-1)*blockDim().y
  if (b>size(ba,1) || ak>size(ba,2))
    return nothing
  end
  T = size(ei,3)
  R = eltype(ba)
  xx = zeros(MMatrix{P+2,P+2,R})
  xy = zeros(MVector{P+2,R})
  xt = zeros(MVector{P+2,R})
  xt[1] = one(R)
  yy = zero(R)
  xx[1,1] = T-P
  xx[1,2] = xx[2,1] = (T+1)*T/2 - (P+1)*P/2 #sum((P+1):T)
  xx[2,2] = (2*T+1)*T*(T+1)/6 - (2*P+1)*P*(P+1)/6 #sum((P+1:T).^2)
  α = zeros(MVector{P+2, R})
  α[3] = αgrid[ak]
  @inbounds for t = (P+1):T
    xt[2] = t
    for i = 1:(P+2)
      for j = 3:(P+2)
        xx[i,j] += xt[i]*xt[j]
      end
    end
    y = e[ei[b,ak,t]] + dot(α,xt)
    for i in 1:(P+2)
      xy[i] += xt[i]*y
    end
    yy += y*y
    for i = 4:(P+2) # shift y lags
      xt[i] = xt[i-1]
    end
    xt[3] = y
  end

  for i = 3:(P+2)
     for j = 1:(i-1)
       xx[i,j] = xx[j,i]
     end
  end
  ixx = inv(xx)
  θ3 = zero(R)
  for i = 1:3
    θ3 += ixx[3,i]*xy[i]
  end
  ee = yy
  for i = 1:3
    for j = 1:3
      ee -= xy[i]*ixx[i,j]*xy[j]
    end
  end
  se3 = CUDA.sqrt(ixx[3,3]*ee/(T-(2*P+2)))
  bootq[b,ak]  = θ3
  bootse[b,ak] = se3
  ba[b,ak] = bootq[b,ak] - αgrid[ak]
  return nothing
end
