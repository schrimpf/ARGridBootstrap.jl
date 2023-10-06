"""
    ar1_original(y0, a, e, rindex=T->rand(1:length(e), T))

Simulate AR1 model by sampling errors from e with replacement.

    y[t] = a*y[t-1] + ϵ[t]

# Arguments
- `y0`: initial value for `y`
- `a`: AR parameter
- `e`: values of for error term. `ϵ = e[rindex(T)]]`
- `rindex` function that returns random index in 1:length(e)

# Returns
- `y`: vector of length `T = length(e)`
"""
function ar1_original(y0, a, e, rindex=T->rand(1:length(e),T))
  T = length(e)
  y = Array{eltype(e)}(undef, T)
  y[1] = abs(a)<1 ? y0 : zero(eltype(y))
  et = e[rindex(T-1)]
  for t in 2:T
    y[t] = a*y[t-1] + et[t-1]
  end
  y
end


"""
    b_est_original(y)

Estimate AR(1) model with intercept and time trend

    y[t] = θ[0] + θ[1]t + θ[2]y[t-1] + e[t]

# Arguments
- `y`: vector

# Returns
- `θ`: estimated coefficients
- `se`: standard errors
- `e`: residuals
"""
function b_est_original(yin)
  T = length(yin)
  x = [ones(T-1) 2:T yin[1:(T-1)]]
  y = yin[2:T]
  θ = x'*x \ x'y
  e = y - x*θ
  se = sqrt.(diag(inv(x'*x) *(e'*e))./(T-4))
  (θ=θ,se=se,e=e)
end

function b_est_prealloc!(x, yin)
  T = length(yin)
  x[:,1] .= 1
  x[:,2] .= 2:T
  x[:,3] .= yin[1:(end-1)]
  y = yin[2:T]
  θ = x'*x \ x'y
  e = y - x*θ
  se = sqrt.(diag(inv(x'*x) *(e'*e))./(T-4))
  (θ=θ,se=se,e=e)
end


"""
    b_est_mldivide(y)

  Estimate AR(1) model with intercept and time trend.

    y[t] = θ[0] + θ[1]t + θ[2]y[t-1] + e[t]

# Arguments
- `y`: vector

# Returns
- `θ`: estimated coefficients
- `se`: standard errors
- `e`: residuals
"""
function b_est_mldivide(yin)
  T = length(yin)
  x = [ones(T-1) 2:T yin[1:(T-1)]]
  y = yin[2:T]
  tmp = x'*x \ [x'*y I]
  θ = tmp[:,1]
  ixx = tmp[:,2:4]
  e = y - x*θ
  se = sqrt.(diag(ixx *(e'*e))./(T-4))
  (θ=θ,se=se,e=e)
end

"""
    b_est_nox(y)

  Estimate AR(1) model with intercept and time trend.

    y[t] = θ[0] + θ[1]t + θ[2]y[t-1] + e[t]

# Arguments
- `y`: vector

# Returns
- `θ`: estimated coefficients
- `se`: standard errors
- `e`: residualas
"""
function b_est_nox(yin)
  T = length(yin)
  xx = @MMatrix zeros(eltype(yin),3,3)
  xy = @MVector zeros(eltype(yin),3)
  @inbounds @simd for t in 2:T
    xx[1,3] += yin[t-1]
    #xx[2,3] += t*yin[t-1]
    xx[2,3] = muladd(t,yin[t-1], xx[2,3])
    xx[3,3] += yin[t-1]^2
    xy[1] += yin[t]
    #xy[2] += t*yin[t]
    xy[2] = muladd(t, yin[t], xy[2])
    xy[3] = muladd(yin[t-1],yin[t], xy[3])
  end
  xx[1,1] = T-1 # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - 1 # sum(p+1:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - 1 # sum((p+1:T).^2)
  xx[3,1] = xx[1,3]
  xx[3,2] = xx[2,3]
  ixx = inv(xx)
  θ = ixx * xy
  e = similar(yin,T-1)
  @simd for t in 2:T
    @inbounds e[t-1] = yin[t] - θ[1] - θ[2]*t - θ[3]*yin[t-1]
  end
  se = sqrt.(diag(ixx *(e'*e))./(T-4))
  (θ=θ,se=se,e=e)
end

"""
    simulate_estimate_arp(y0, a, e, ar::Val{P}, rindex=T->rand(1:length(e),T))

Simulates and estimates an AR(P) model. `y` is simulated as

   y[t] = a*y[t-1] + ϵ[t]

and the estimate of θ from

   y[t] = θ[1] + θ[2]t + θ[3] y[t-1] + ... + θ[P] y[t-P] + u[t]

is computed.

# Arguments
- `y0` initial value of y
- `a` AR(1) parameter
- `e` error terms to sample from `ϵ[t] = e[rindex(1)]`
- `ar::Val{P}` order of autoregressive model to estimate
- `rindex` function that returns random index in 1:length(e)

# Returns
- `θ` estimated coefficients
- `se` standard errors
"""
function simulate_estimate_arp(y0, a, e, ar::Val{P}=Val(1),
                               rindex=()->rand(1:length(e))) where P
  T = length(e)
  length(a)==P || error("length(a) not equal to P")
  xx = @MMatrix zeros(eltype(e),P+2, P+2)
  xy = @MVector zeros(eltype(e),P+2)
  yy = zero(eltype(e))
  xt = @MVector ones(eltype(e), P+2)
  if (abs(a)<1)
    xt[3:(P+2)] .= y0
  else
    xt[3:(P+2)] .= 0.0
  end
  α = @MVector zeros(eltype(e),P+2)
  @simd for i = 1:P
    α[2+i] = a[i]
  end

  xx[1,1] = T-P # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - sum(1:P) # sum(P+1:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - sum((1:P).^2) # sum((P+1:T).^2)
  @inbounds for t in (P+1):T
    et = e[rindex()]
    xt[2] = t
    for i in 1:(P+2)
      @simd for j in 3:(P+2)
        xx[i,j] += xt[i]*xt[j]
      end
    end
    y = dot(α, xt) + et
    @simd for i in 1:(P+2)
      xy[i] += xt[i]*y
    end
    yy += y^2
    if (P>1)
      xt[4:(P+2)] .= xt[3:(P+1)]
    end
    xt[3] = y
  end
  @inbounds for i in 3:(P+2)
    for j in 1:(i-1)
      xx[i,j] = xx[j,i]
    end
  end
  ixx = inv(xx)
  θ = ixx*xy
  ee = yy - xy'*ixx*xy
  se = sqrt.(abs.(diag(ixx *(ee))./(T-(2*P+2))))
  (θ=θ,se=se)
end

"""
    simulate_estimate_arp_lv(y0, a, e, ar::Val{P}, rindex=T->rand(1:length(e),T))

Simulates and estimates an AR(P) model. Uses LoopVectorization.jl to
produce fast code.

`y` is simulated as

   y[t] = a*y[t-1] + ϵ[t]

and the estimate of θ from

   y[t] = θ[1] + θ[2]t + θ[3] y[t-1] + ... + θ[P] y[t-P] + u[t]

is computed.

# Arguments
- `y0` initial value of y
- `a` AR(1) parameter
- `e` error terms to sample from `ϵ[t] = e[rindex(1)]`
- `ar::Val{P}` order of autoregressive model to estimate
- `rindex` function that returns random index in 1:length(e)

# Returns
- `θ` estimated coefficients
- `se` standard errors
"""
function simulate_estimate_arp_lv(y0, a, e, ar::Val{P}=Val(1),
                               rindex=()->rand(1:length(e))) where P
  T = length(e)
  length(a)==P || error("length(a) not equal to P")
  xx = @MMatrix zeros(eltype(e),P+2, P+2)
  xy = @MVector zeros(eltype(e),P+2)
  yy = zero(eltype(e))
  xt = @MVector ones(eltype(e), P+2)
  if (abs(a)<1)
    xt[3:(P+2)] .= y0
  else
    xt[3:(P+2)] .= 0.0
  end
  α = @MVector zeros(eltype(e),P+2)
  @simd for i = 1:P
    α[2+i] = a[i]
  end

  xx[1,1] = T-P # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - sum(1:P) # sum(P+1:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - sum((1:P).^2) # sum((P+1:T).^2)
  @inbounds for t in (P+1):T
    et = e[rindex()]
    xt[2] = t
    @turbo for i in 1:(P+2)
      for j in 3:(P+2)
        xx[i,j] += xt[i]*xt[j]
      end
    end
    y = dot(α, xt) + et
    @turbo for i in 1:(P+2)
      xy[i] += xt[i]*y
    end
    yy += y^2
    if (P>1)
      xt[4:(P+2)] .= xt[3:(P+1)]
    end
    xt[3] = y
  end
  @inbounds for i in 3:(P+2)
    for j in 1:(i-1)
      xx[i,j] = xx[j,i]
    end
  end
  ixx = inv(xx)
  θ = ixx*xy
  ee = yy - xy'*ixx*xy
  se = sqrt.(abs.(diag(ixx *(ee))./(T-(2*P+2))))
  (θ=θ,se=se)
end
