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
    ?? # discount factor
    ???# agents
    ????# state space
    ???? # joint action space
    ???? # joint observation space
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
    (??::ConditionalPlan)() = ??.a
    (??::ConditionalPlan)(o) = ??.subplans[o]




#const policyAgent1= Dict(
#    "quiet"=>ConditionalPlan("sing"), "crying"=>ConditionalPlan("ignore")
#)
#ConditionalPlan("ignore",policyAgent1)


joint(X) = vec(collect(Iterators.product(X...)))


function lookahead(????::POMG, U, s, a)
    ????, ????, T, O, R, ?? = ????.????, joint(????.????), ????.T, ????.O, ????.R, ????.??
    u??? = sum(T(s,a,s???)*sum(O(a,s???,o)*U(o,s???) for o in ????) for s??? in ????)
    return R(s,a) + ??*u???
end

function evaluate_plan(????::POMG, ??, s)
    a = Tuple(??i() for ??i in ??)
    U(o,s???) = evaluate_plan(????, [??i(oi) for (??i, oi) in zip(??,o)], s???)
    return isempty(first(??).subplans) ? ????.R(s,a) : lookahead(????, U, s, a)
end
function utility(????::POMG, b, ??)
    u = [evaluate_plan(????, ??, s) for s in ????.????]
    return sum(bs * us for (bs, us) in zip(b, u))
end

p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
b=[0.5,0.5];
??1= Dict(
    "crying"=>ConditionalPlan("feed"), "quiet"=>ConditionalPlan("ignore")
);
??2= Dict(
    "quiet"=>ConditionalPlan("feed"), "crying"=>ConditionalPlan("sing")
);

??s= [ConditionalPlan("ignore",??1),ConditionalPlan("ignore",??2)];
#ConditionalPlan("ignore",policyAgent1)
#print(first(??s).subplans)
utility(p,b,??s)
#-------------------------------------------

#---------------------------------------
struct SimpleGame
    ??     # discount factor
    ???     # agents
    ????     # joint action space
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
(??i::SimpleGamePolicy)(ai) = get(??i.p, ai, 0.0)

    

struct NashEquilibrium end
function tensorform(????::SimpleGame)
    ???, ????, R = ????.???, ????.????, ????.R
    ?????? = eachindex(???)
    ??????? = [eachindex(????[i]) for i in ???]
    R??? = [R(a) for a in joint(????)]
    return ??????, ???????, R???
end
  
function solve(M::NashEquilibrium, ????::SimpleGame)
    ???, ????, R = tensorform(????)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[???])
    @variable(model, ??[i=???, ????[i]] ??? 0)
    @NLobjective(model, Min,
      sum(U[i] - sum(prod(??[j,a[j]] for j in ???) * R[y][i]
        for (y,a) in enumerate(joint(????))) for i in ???))
    @NLconstraint(model, [i=???, ai=????[i]],
      U[i] ??? sum(
        prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : ??[j,a[j]] for j in ???)
        * R[y][i] for (y,a) in enumerate(joint(????))))
    @constraint(model, [i=???], sum(??[i,ai] for ai in ????[i]) == 1)
    optimize!(model)
    ??i???(i) = SimpleGamePolicy(????.????[i][ai] => value(??[i,ai]) for ai in ????[i])
    return [??i???(i) for i in ???]
end

struct POMGNashEquilibrium
    b # initial belief
    d # depth of conditional plans
end
function create_conditional_plans(????, d)
    ???, ????, ???? = ????.???, ????.????, ????.????
    ?? = [[ConditionalPlan(ai) for ai in ????[i]] for i in ???]
    for t in 1:d
        ?? = expand_conditional_plans(????, ??)
    end
    return ??
end
function expand_conditional_plans(????, ??)
    ???, ????, ???? = ????.???, ????.????, ????.????
    return [[ConditionalPlan(ai, Dict(oi => ??i for oi in ????[i]))
        for ??i in ??[i] for ai in ????[i]] for i in ???]
end
function solve(M::POMGNashEquilibrium, ????::POMG)
    ???, ??, b, d = ????.???, ????.??, M.b, M.d
    ?? = create_conditional_plans(????, d)
    U = Dict(?? => utility(????, b, ??) for ?? in joint(??))
    ???? = SimpleGame(??, ???, ??, ?? -> U[??])
    ?? = solve(NashEquilibrium(), ????)
    return Tuple(argmax(??i.p) for ??i in ??)
end

#p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
#b=[0.5,0.5];
#dyP=POMGNashEquilibrium(b,2);
#print(solve(dyP,p));

#----------------------------
struct SimpleGame
    ??     # discount factor
    ???     # agents
    ????     # joint action space
    R       # joint reward funct
    function SimpleGame(discount, agents, jointActionSpace, jointRewardFunc)
      new(discount, agents, jointActionSpace, jointRewardFunc)
    end
  end




struct NashEquilibrium end
function tensorform(????::SimpleGame)
    ???, ????, R = ????.???, ????.????, ????.R
    ?????? = eachindex(???)
    ??????? = [eachindex(????[i]) for i in ???]
    R??? = [R(a) for a in joint(????)]
    return ??????, ???????, R???
end
  
function solve(M::NashEquilibrium, ????::SimpleGame)
    ???, ????, R = tensorform(????)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[???])
    @variable(model, ??[i=???, ????[i]] ??? 0)
    @NLobjective(model, Min,
      sum(U[i] - sum(prod(??[j,a[j]] for j in ???) * R[y][i]
        for (y,a) in enumerate(joint(????))) for i in ???))
    @NLconstraint(model, [i=???, ai=????[i]],
      U[i] ??? sum(
        prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : ??[j,a[j]] for j in ???)
        * R[y][i] for (y,a) in enumerate(joint(????))))
    @constraint(model, [i=???], sum(??[i,ai] for ai in ????[i]) == 1)
    optimize!(model)
    ??i???(i) = SimpleGamePolicy(????.????[i][ai] => value(??[i,ai]) for ai in ????[i])
    return [??i???(i) for i in ???]
end
  

function expand_conditional_plans(????, ??)
    ???, ????, ???? = ????.???, ????.????, ????.????
    return [[ConditionalPlan(ai, Dict(oi => ??i for oi in ????[i]))
        for ??i in ??[i] for ai in ????[i]] for i in ???]
end


struct POMGDynamicProgramming
    b # initial belief
    d # depth of conditional plans
end
function solve(M::POMGDynamicProgramming, ????::POMG)
    ???, ????, ????, R, ??, b, d = ????.???, ????.????, ????.????, ????.R, ????.??, M.b, M.d
    ?? = [[ConditionalPlan(ai) for ai in ????[i]] for i in ???]
    for t in 1:d
        ?? = expand_conditional_plans(????, ??)
        prune_dominated!(??, ????)
    end
    ???? = SimpleGame(??, ???, ??, ?? -> utility(????, b, ??))
    ?? = solve(NashEquilibrium(), ????)
    return Tuple(argmax(??i.p) for ??i in ??)
end

function prune_dominated!(??, ????::POMG)
    done = false
    while !done
        done = true
        for i in shuffle(????.???)
            for ??i in shuffle(??[i])
                if length(??[i]) > 1 && is_dominated(????, ??, i, ??i)
                    filter!(??i??? -> ??i??? ??? ??i, ??[i])
                    done = false
                    break
                end
            end
        end
    end
end

function is_dominated(????::POMG, ??, i, ??i)
    ???, ???? = ????.???, ????.????
    joint??noti = joint([??[j] for j in ??? if j ??? i])
    ??(??i???, ??noti) = [j==i ? ??i??? : ??noti[j>i ? j-1 : j] for j in ???]
    Ui = Dict((??i???, ??noti, s) => evaluate_plan(????, ??(??i???, ??noti), s)[i]
            for ??i??? in ??[i], ??noti in joint??noti, s in ????)
    model = Model(Ipopt.Optimizer)
    @variable(model, ??)
    @variable(model, b[joint??noti, ????] ??? 0)
    @objective(model, Max, ??)
    @constraint(model, [??i???=??[i]],
        sum(b[??noti, s] * (Ui[??i???, ??noti, s] - Ui[??i, ??noti, s])
        for ??noti in joint??noti for s in ????) ??? ??)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(??) ??? 0
end
   

p=POMG(0.9,[1,2],["hungry","sated"],[["feed","sing","ignore"],["feed","sing","ignore"]],[["crying","quiet"],["crying","quiet"]],Transition,Observation,Reward);
b=[0.5,0.5];
dyP=POMGDynamicProgramming(b,2);
print(solve(dyP,p));










