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
  
end

struct bEstCached
  T::Int64
  x::Matrix{Float64}
  y::Vector{Float64}
  xx::Matrix{Float64}
end
function bEstCached(T::Int64)
  x=ones(T-1, 3)
  x[:,2].=2:T
  y=zeros(T-1)
  xx=zeros(3,3)
  bEstCached(T,x,y,xx)
end
function (cache::bEstCached)(yin)
  @unpack T, x, y,xx = cache
  @assert length(yin)==T
  @views x[:,3] .= yin[1:(end-1)]
  @views y .= yin[2:end]
  xx .= x'*x
  cxx = cholesky(xx)
  θ = cxx \ x'*y
  e = y - x*θ
  se = sqrt.(diag(inv(cxx) *(e'*e))./(T-4))
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
function b_est_nox(yin; xx_xy!::F=xx_xy!, resids!::FR=resids!) where {F<:Function, FR<:Function}
  T = length(yin)
  xx = @MMatrix zeros(eltype(yin),3,3)
  xy = @MVector zeros(eltype(yin),3)
  xx_xy!(xx,xy,yin)
  ixx = inv(xx)
  θ = ixx * xy
  e = similar(yin,T-1)
  resids!(e,yin,θ)
  se = sqrt.(diag(ixx *(e'*e))./(T-4))
  (θ=θ,se=se,e=e)
end 

@inline function resids!(e, yin, θ)
  T = length(yin)
  @inbounds @simd for t in 2:T
    e[t-1] = yin[t] - θ[1] - θ[2]*t - θ[3]*yin[t-1]
  end
  nothing
end

@inline function resids_turbo!(e, yin, θ)
  T = length(yin)
  @turbo for t in 2:T
    e[t-1] = yin[t] - θ[1] - θ[2]*t - θ[3]*yin[t-1]
  end
  nothing
end

"""
   oneto(Val(N))
Creates at a compile time the tuple (1, 2, ..., N)
"""
oneto(::Val{1}) = (1,)
oneto(::Val{N}) where N = (oneto(Val(N-1))..., N)

@inline function resids_simd!(e,yin, θ, width::Val{N}=Val(8)) where N
  lane = VecRange{N}(0)
  tv=Vec{N,Float64}(oneto(Val(N)))
  θ1=-Vec{N,Float64}(θ[1])
  θ2=-Vec{N,Float64}(θ[2])
  θ3=-Vec{N,Float64}(θ[3])
  remainder = length(e) % N
  @inbounds for t ∈ 1:N:(length(e)-remainder) #eachindex(e)
    @fastmath e[t+lane] = muladd(θ2,tv,yin[t+1+lane])+muladd(θ3,yin[t+lane],θ1)
    @fastmath tv+=N
  end
  @inbounds for t ∈ (length(e)-remainder+1):length(e)
    @fastmath e[t] = muladd(-θ[2],t+1,yin[t+1])-muladd(θ[3],yin[t],θ[1])
  end
  nothing
end
  
@inline function xx_xy!(xx,xy,yin)
  T = length(yin)
  xx .= zero(eltype(xx))
  xy .= zero(eltype(xy))
  @inbounds @simd for t in 2:T
    xx[1,3] += yin[t-1]
    xx[2,3] += t*yin[t-1]
    xx[3,3] += yin[t-1]^2
    xy[1] += yin[t]
    xy[2] += t*yin[t]
    xy[3] += yin[t-1]*yin[t]
  end 
  xx[1,1] = T-1 # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - 1 # sum(p+1:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - 1 # sum((p+1:T).^2)  
  xx[3,1] = xx[1,3]
  xx[3,2] = xx[2,3]
  nothing
end

@inline function xx_xy_simd!(xx,xy,yin, v::Val{N}=Val(32)) where {N}
  T = length(yin)
  remainder=(T-1) % N
  xx .= zero(eltype(xx))
  xy .= zero(eltype(xy))
  tv = Vec{N,eltype(yin)}(oneto(Val(N)))+1
  lane = VecRange{N}(0)
  @inbounds for t in 2:N:(T-remainder)
    xx[1,3] += sum(yin[t-1+lane])
    xx[2,3] += sum(yin[t-1+lane]*tv)
    xx[3,3] += sum(yin[t-1+lane]^2)
    xy[1] += sum(yin[t+lane])
    xy[2] += sum(tv*yin[t+lane])
    xy[3] += sum(yin[t-1+lane]*yin[t+lane])
    tv += N
  end
  @inbounds for t in (T-remainder+1):T
    xx[1,3] += yin[t-1]
    xx[2,3] += yin[t-1]*t
    xx[3,3] += yin[t-1]^2
    xy[1] += yin[t]
    xy[2] += t*yin[t]
    xy[3] += yin[t-1]*yin[t]
  end
  xx[1,1] = T-1 # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - 1 # sum(2:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - 1 # sum((2:T).^2)
  xx[3,1] = xx[1,3]
  xx[3,2] = xx[2,3]
  nothing
end

""" 
    b_est_stride(y)

  Estimate AR(1) model with intercept and time trend. 

    y[t] = θ[0] + θ[1]t + θ[2]y[t-1] + e[t]

# Arguments
- `y`: vector 

# Returns
- `θ`: estimated coefficients
- `se`: standard errors
- `e`: residualas 
"""
function b_est_stride(yin)
  T = length(yin)
  xx = StrideArray{eltype(yin)}(undef, StaticInt(3), StaticInt(3))
  @turbo xx .= 0.0
  xy = StrideArray{eltype(yin)}(undef, StaticInt(3))
  θ = StrideArray{eltype(yin)}(undef, StaticInt(3)) #similar(xy)
  se = similar(xy)
  @turbo xy .= 0.0
  xx13 = zero(eltype(yin))
  xx23 = zero(eltype(yin))
  xx33 = zero(eltype(yin))
  xy1 = zero(eltype(yin))
  xy2 = zero(eltype(yin))
  xy3 = zero(eltype(yin))
  @turbo for t ∈ 2:T
    ylag = yin[t-1]
    y = yin[t] 
    xx13 += ylag
    xx23 += t*ylag
    xx33 += ylag^2
    #xy1 += y
    xy2 += t*y
    xy3 += ylag*y
  end
  xy1 = xx13 - yin[1] + yin[T]
  xx[1,3] = xx13
  xx[2,3] = xx23
  xx[3,3] = xx33
  xy[1]=xy1
  xy[2]=xy2
  xy[3]=xy3
  xx[1,1] = T-1 # = 1'*1
  xx[1,2] = xx[2,1] = (T+1)*T/2 - 1 # sum(p+1:T)
  xx[2,2] = (2*(T)+1)*(T)*(T+1)/6 - 1 # sum((p+1:T).^2)  
  xx[3,1] = xx[1,3]
  xx[3,2] = xx[2,3]
  ixx = invsym(xx)
  
  @turbo for i ∈ eachindex(θ)
    tij = zero(eltype(θ))
    for j ∈ axes(ixx,2)
      tij += ixx[i,j]*xy[j]
    end
    θ[i] = tij
  end

  e = similar(yin,T-1)
  @turbo for t in 2:T
    e[t-1] = yin[t] - θ[1] - θ[2]*t - θ[3]*yin[t-1]
  end
  #@show se = sqrt.(diag(ixx *(e'*e))./(T-4))
  σ² = vsum(x->x^2,e)/(T-4)
  @turbo for j ∈ 1:3
    se[j] = sqrt(ixx[j,j]*σ²) 
  end
  
  (θ=θ,se=se,e=e)
end

"""
fast inverse function for statically sized 3x3 StrideArray
"""
@inline function invsym(A::typeof(StrideArray(undef, StaticInt(3), StaticInt(3))))
  iA = similar(A)
  iA[1,1]=A[3,3]*A[2,2]-A[3,2]*A[2,3]
  iA[1,2]=-(A[3,3]*A[1,2]-A[3,2]*A[1,3])
  iA[1,3]=A[2,3]*A[1,2]-A[2,2]*A[1,3]
  iA[2,1]=iA[1,2] #-A[3,3]*A[2,1]-A[3,1]*A[2,3]
  iA[2,2]=A[3,3]*A[1,1]-A[3,1]*A[1,3]
  iA[2,3]=-(A[2,3]*A[1,1]-A[2,1]*A[1,3])
  iA[3,1]=iA[1,3]
  iA[3,2]=iA[2,3]
  iA[3,3]=A[2,2]*A[1,1]-A[2,1]*A[1,2]
  det = A[1,1]*iA[1,1]+A[2,1]*iA[1,2]+A[3,1]*iA[1,3]
  @turbo iA ./= det
  return(iA)
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
