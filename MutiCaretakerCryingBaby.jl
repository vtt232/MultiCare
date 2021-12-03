using Distributions

using JuMP
using GLPK
using Ipopt
using DataStructures
using Random

function Base.findmax(f::Function, xs)
    f_max = -Inf
    x_max = first(xs)
    for x in xs
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

Base.argmax(f::Function, xs) = findmax(f, xs)[2]
    
Base.Dict{Symbol,V}(a::NamedTuple) where V =
    Dict{Symbol,V}(n=>v for (n,v) in zip(keys(a), values(a)))


struct DiscountFactor
    y::Float64

    function DiscountFactor()
        new(0.9)
    end
end

struct Agent
    a::Vector{Int64}

    function Agent()
        x=[1,2]
        new(x)
    end
end

struct StateSpace
    stateSpace::Vector{String}

    function StateSpace()
        x=["hungry","sated"]
        new(x)
        
    end
end

struct ActionSpace
    actionSpace::Vector{String}

    function ActionSpace()
        x=["feed","sing","ignore"]
        new(x)
    end
end

struct ObservationSpace
    observationSpace::Vector{String}

    function ObservationSpace()
        x=["crying","quiet"]
        new(x)
    end
end 

function Transition(fromState, actions::Tuple{String,String}, toState )
    if fromState=="hungry" && toState=="sated" && (actions[1]=="feed" || actions[2]=="feed") 
        return 1;
    elseif fromState=="hungry" && toState=="sated" && (actions[1]!="feed" && actions[2]!="feed")
        return 0.5;
    elseif fromState=="hungry" && toState=="hungry"&& (actions[1]!="feed" && actions[2]!="feed")
        return 1;
    elseif fromState=="hungry" && toState=="hungry"&& (actions[1]=="feed" || actions[2]=="feed")
        return 0;
    elseif fromState=="sated" && toState=="sated" && (actions[1]=="feed" || actions[2]=="feed")
        return 1;
    elseif fromState=="sated" && toState=="sated" && (actions[1]!="feed" && actions[2]!="feed")
        return 0.9;
    elseif fromState=="sated" && toState=="hungry" && (actions[1]!="feed" && actions[2]!="feed")
        return 0.1;
    elseif fromState=="sated" && toState=="hungry" && (actions[1]=="feed" || actions[2]=="feed")
        return 0;
    else return 0;
    end
end

#Transition("hungry",("sing","feed"),"hungry")
    
function Observation(actions::Tuple{String,String},toState::String,observes::Tuple{String, String})
    if (observes[1]=="crying"&&observes[2]=="crying") && (actions[1]=="sing"||actions[2]=="sing")&&toState=="hungry"
        return 0.9;
    elseif (observes[1]=="quiet"&&observes[2]=="quiet") && (actions[1]=="sing"||actions[2]=="sing")&&toState=="hungry"
        return 0.1;
    elseif (observes[1]=="crying"&&observes[2]=="crying") && (actions[1]=="sing"||actions[2]=="sing")&&toState=="sated"
        return 0;
    elseif (observes[1]=="quiet"&&observes[2]=="quiet") && (actions[1]=="sing"||actions[2]=="sing")&&toState=="sated"
        return 1;
    elseif (observes[1]=="crying"&&observes[2]=="crying") && (actions[1]!="sing"&&actions[2]!="sing")&&toState=="hungry"
        return 0.9;
    elseif (observes[1]=="quiet"&&observes[2]=="quiet") && (actions[1]!="sing"&&actions[2]!="sing")&&toState=="hungry"
        return 0.1;
    elseif (observes[1]=="crying"&&observes[2]=="crying") && (actions[1]!="sing"&&actions[2]!="sing")&&toState=="sated"
        return 0;
    elseif (observes[1]=="quiet"&&observes[2]=="quiet") && (actions[1]!="sing"&&actions[2]!="sing")&&toState=="sated"
        return 1;
    else return 0;
    end
end

function Reward(state,actions::Tuple{String,String})
    r1=0.0;
    r2=0.0;
    if state=="hungry"
        r1=r1-10.0;
        r2=r2-10.0;
    end
    if actions[1]=="feed"
        r1=r1-2.5;
    end
    if actions[2]=="feed"
        r2=r2-5.0;
    end
    if actions[1]=="sing"
        r1=r1-0.5;
    end
    if actions[2]=="sing"
        r2=r2-0.25;
    end

    return [r1,r2];
end

#r=Reward("hungry",("ignore","ignore"))




struct POMG
    γ # discount factor
    ℐ# agents
    𝒮# state space
    𝒜 # joint action space
    𝒪 # joint observation space
    T # transition function
    O # joint observation function
    R # joint reward function
    function POMG(discount,agents,states,jointAction,jointObservation,transitionFunction,jointObservationFunction,jointRewardFunction)
        new(discount,agents,states,jointAction,jointObservation,transitionFunction,jointObservationFunction,jointRewardFunction);
    end
end

#m=POMG();
#print(m);

struct ConditionalPlan
    a # action to take at root
    subplans # dictionary mapping observations to subplans
end
    ConditionalPlan(a) = ConditionalPlan(a, Dict())
    (π::ConditionalPlan)() = π.a
    (π::ConditionalPlan)(o) = π.subplans[o]




#const policyAgent1= Dict(
#    "quiet"=>ConditionalPlan("sing"), "crying"=>ConditionalPlan("ignore")
#)
#ConditionalPlan("ignore",policyAgent1)


joint(X) = vec(collect(Iterators.product(X...)))


function lookahead(𝒫::POMG, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    u′ = sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return R(s,a) + γ*u′
end

function evaluate_plan(𝒫::POMG, π, s)
    a = Tuple(πi() for πi in π)
    U(o,s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π,o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫, U, s, a)
end
function utility(𝒫::POMG, b, π)
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u))
end

p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
b=[0.5,0.5];
π1= Dict(
    "crying"=>ConditionalPlan("feed"), "quiet"=>ConditionalPlan("ignore")
);
π2= Dict(
    "quiet"=>ConditionalPlan("feed"), "crying"=>ConditionalPlan("sing")
);

πs= [ConditionalPlan("ignore",π1),ConditionalPlan("ignore",π2)];
#ConditionalPlan("ignore",policyAgent1)
#print(first(πs).subplans)
utility(p,b,πs)
#-------------------------------------------

#---------------------------------------
struct SimpleGame
    γ     # discount factor
    ℐ     # agents
    𝒜     # joint action space
    R       # joint reward funct
    function SimpleGame(discount, agents, jointActionSpace, jointRewardFunc)
      new(discount, agents, jointActionSpace, jointRewardFunc)
    end
end

struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end
    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
    return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end
    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end
(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)

    

struct NashEquilibrium end
function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end
  
function solve(M::NashEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[ℐ])
    @variable(model, π[i=ℐ, 𝒜[i]] ≥ 0)
    @NLobjective(model, Min,
      sum(U[i] - sum(prod(π[j,a[j]] for j in ℐ) * R[y][i]
        for (y,a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i=ℐ, ai=𝒜[i]],
      U[i] ≥ sum(
        prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : π[j,a[j]] for j in ℐ)
        * R[y][i] for (y,a) in enumerate(joint(𝒜))))
    @constraint(model, [i=ℐ], sum(π[i,ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i,ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end

struct POMGNashEquilibrium
    b # initial belief
    d # depth of conditional plans
end
function create_conditional_plans(𝒫, d)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
    end
    return Π
end
function expand_conditional_plans(𝒫, Π)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i]))
        for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end
function solve(M::POMGNashEquilibrium, 𝒫::POMG)
    ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
    Π = create_conditional_plans(𝒫, d)
    U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
    𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

#p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
#b=[0.5,0.5];
#dyP=POMGNashEquilibrium(b,2);
#print(solve(dyP,p));

#----------------------------
struct SimpleGame
    γ     # discount factor
    ℐ     # agents
    𝒜     # joint action space
    R       # joint reward funct
    function SimpleGame(discount, agents, jointActionSpace, jointRewardFunc)
      new(discount, agents, jointActionSpace, jointRewardFunc)
    end
  end




struct NashEquilibrium end
function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end
  
function solve(M::NashEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[ℐ])
    @variable(model, π[i=ℐ, 𝒜[i]] ≥ 0)
    @NLobjective(model, Min,
      sum(U[i] - sum(prod(π[j,a[j]] for j in ℐ) * R[y][i]
        for (y,a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i=ℐ, ai=𝒜[i]],
      U[i] ≥ sum(
        prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : π[j,a[j]] for j in ℐ)
        * R[y][i] for (y,a) in enumerate(joint(𝒜))))
    @constraint(model, [i=ℐ], sum(π[i,ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i,ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end
  

function expand_conditional_plans(𝒫, Π)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i]))
        for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end


struct POMGDynamicProgramming
    b # initial belief
    d # depth of conditional plans
end
function solve(M::POMGDynamicProgramming, 𝒫::POMG)
    ℐ, 𝒮, 𝒜, R, γ, b, d = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ, M.b, M.d
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
        prune_dominated!(Π, 𝒫)
    end
    𝒢 = SimpleGame(γ, ℐ, Π, π -> utility(𝒫, b, π))
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

function prune_dominated!(Π, 𝒫::POMG)
    done = false
    while !done
        done = true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i]) > 1 && is_dominated(𝒫, Π, i, πi)
                    filter!(πi′ -> πi′ ≠ πi, Π[i])
                    done = false
                    break
                end
            end
        end
    end
end

function is_dominated(𝒫::POMG, Π, i, πi)
    ℐ, 𝒮 = 𝒫.ℐ, 𝒫.𝒮
    jointΠnoti = joint([Π[j] for j in ℐ if j ≠ i])
    π(πi′, πnoti) = [j==i ? πi′ : πnoti[j>i ? j-1 : j] for j in ℐ]
    Ui = Dict((πi′, πnoti, s) => evaluate_plan(𝒫, π(πi′, πnoti), s)[i]
            for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    model = Model(Ipopt.Optimizer)
    @variable(model, δ)
    @variable(model, b[jointΠnoti, 𝒮] ≥ 0)
    @objective(model, Max, δ)
    @constraint(model, [πi′=Π[i]],
        sum(b[πnoti, s] * (Ui[πi′, πnoti, s] - Ui[πi, πnoti, s])
        for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(δ) ≥ 0
end
   

p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
b=[0.5,0.5];
dyP=POMGDynamicProgramming(b,2);
print(solve(dyP,p));










