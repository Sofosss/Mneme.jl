using PythonCall

np = pyimport("numpy")

using DataFrames, CSV

import ..torcjulia

mutable struct Pipeline
    operators::Vector{Union{MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder}}
    file::String

end

Pipeline(operators::Vector{Union{MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder}}, file::String) =
    Pipeline(operators, file)


abstract type OpDisp end

struct StandardDisp <: OpDisp
    with_std::Bool
end
struct MaxAbsDisp <: OpDisp end
struct MinMaxDisp <: OpDisp end
struct OrdinalDisp <: OpDisp end
struct OneHotDisp <: OpDisp end
struct LabelDisp <: OpDisp end

function fit(pipeline::Pipeline, reader::BlockReader)
    offsets = reader.block_offsets

    _mask = _parse_ops(pipeline, reader); _act_ops = pipeline.operators[_mask]

    _features = _get_union_features(_act_ops, reader.columns)
    _feat_mapping = _map_features(_features, reader.feature_idxs_map)
    _feat_dict = Dict(f => i for (i, f) in pairs(_feat_mapping))

    _operators_mapping = [
        [
            get(_feat_dict, f, nothing)
            for f in _map_features(
                isa(operator.features, AbstractVector) ? operator.features : [operator.features],
                reader.feature_idxs_map
            )
        ]
        for operator in _act_ops
    ]

    for (i, op) in enumerate(pipeline.operators)
        op.feature_idxs = _operators_mapping[i]
    end
    
    file = pipeline.file; _disp = _disp_ops(_act_ops)
    args = (_disp, file, _feat_mapping, _operators_mapping)

    offset_chunks = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    
    _partial_res = torcjulia.map(
        _partial_fit,
        offset_chunks,
        chunksize = 1,
        args = args
    )
    
    feature_idxs_map = Dict(
        op => _map_features(
            isa(op.features, AbstractVector) ? op.features : [op.features],
            reader.feature_idxs_map
        ) 
        for op in pipeline.operators
    )

    for (j, i) in enumerate(findall(_mask))
        op = pipeline.operators[i]
        _set_attributes(op, [chunk[j] for chunk in _partial_res];
                    features = op.features,
                    feature_idxs = feature_idxs_map[op])
    end
   
end

function _partial_fit(offsets::Tuple{Int, Int}, operators_disp::Vector{OpDisp}, file::String,
                      feat_mapping::Vector{Int}, _operators_mapping::Vector{<:AbstractVector{<:Integer}})
    chunk = _fetch_chunk(offsets, file, feat_mapping)
    
    [_process_chunk(op_disp, chunk, _operators_mapping[i]) for (i, op_disp) in enumerate(operators_disp)]

end

function _fetch_chunk(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
    open(file, "r") do io
        seek(io, offsets[1])
        buf = offsets[2] === -1 ? read(io) : read(io, offsets[2] - offsets[1])

        CSV.read(
                IOBuffer(buf),
                DataFrame;
                header = false,
                select = feat_mapping,
        )
    end

end

function _get_union_features(operators, columns)
    features = vcat([op.features for op in operators]...)
    unique(col for col in columns if col in features)

end

function _disp_ops(operators)
    map(op -> begin
        if op isa StandardScaler
            StandardDisp(op.params.with_std)
        elseif op isa MaxAbsScaler
            MaxAbsDisp()
        elseif op isa MinMaxScaler
            MinMaxDisp()
        elseif op isa OrdinalEncoder
            OrdinalDisp()
        elseif op isa OneHotEncoder
            OneHotDisp()
        elseif op isa LabelEncoder
            LabelDisp()
        else
            error("Operator of type $(typeof(disp)) is not supported")
        end
    end, operators)

end

function _process_chunk(disp::OpDisp, X::DataFrame, feature_idxs)
    @views cols = eachcol(X)
    @views selected_cols = cols[feature_idxs]
    n_samples = nrow(X)

    if disp isa StandardDisp
        n = length(selected_cols)
        means = Vector{Float64}(undef, n)
        vars  = disp.with_std ? Vector{Float64}(undef, n) : nothing

        @inbounds for i in 1:n
            μ, v = _mean_var(selected_cols[i])
            means[i] = μ
            disp.with_std && (vars[i] = v)
        end
        return means, vars, n_samples

    elseif disp isa MaxAbsDisp
        maxabs_vals = map(maxabs, selected_cols)
        return maxabs_vals, n_samples

    elseif disp isa MinMaxDisp
        n = length(selected_cols)
        mins = Vector{Float64}(undef, n)
        maxs = Vector{Float64}(undef, n)

        @inbounds for i in 1:n
            mn, mx = _minmax(selected_cols[i])
            mins[i] = mn
            maxs[i] = mx
        end
        return mins, maxs, n_samples

    elseif disp isa LabelDisp
        return _unique(selected_cols[1])
    end

    map(_unique, cols[sort(feature_idxs)])
end

function transform(pipeline::Pipeline, X)
    X_transformed = np.array(X)

    for op in pipeline.operators
        _trans_X = transform(op, X)
        py_target_idxs = np.array([i-1 for i in sort(op.feature_idxs)])
        X_transformed[pyslice(nothing), py_target_idxs] =  _trans_X
    end
    
    X_transformed

end

@inline function _mean_var(c)
    μ = zero(eltype(c))
    m2 = zero(eltype(c))
    n = 0
    @inbounds for x in c
        n += 1
        δ = x - μ
        μ += δ / n
        m2 += δ * (x - μ)
    end
    
    μ, m2 / n

end

@inline function _minmax(c)
    mn = typemax(eltype(c))
    mx = typemin(eltype(c))
    @inbounds for x in c
        x < mn && (mn = x)
        x > mx && (mx = x)
    end
    
    mn, mx

end

function _unique(c)
    s = Set{eltype(c)}()
    @inbounds for x in c
        push!(s, x)
    end
    
    collect(s)

end

function _set_attributes(op, stats; features = nothing, feature_idxs = nothing)
    if op isa StandardScaler
        fitted_stats = _reduce_std(stats)
        _set_attributes_std(op.scaler, fitted_stats, features, feature_idxs)
    elseif op isa MaxAbsScaler
        fitted_stats = _reduce_maxabs(stats)
        _set_attributes_maxabs(op.scaler, fitted_stats, features, feature_idxs)
    elseif op isa MinMaxScaler
        fitted_stats = _reduce_mm(stats)
        _set_attributes_mm(op.scaler, fitted_stats, features, feature_idxs)
    elseif op isa OrdinalEncoder
        fitted_stats = _reduce_ord(stats)
        _set_attributes_ord(op.encoder, fitted_stats, features, feature_idxs)
    elseif op isa OneHotEncoder
        fitted_stats = _reduce_ohe(stats)
        _set_attributes_ohe(op.encoder, fitted_stats, features)
    else
        fitted_stats = _reduce_le(stats)
        _set_attributes_le(op.encoder, fitted_stats)
    end

end

function _parse_ops(pipeline, reader)
    _act = trues(length(pipeline.operators))

    for (i, op) in pairs(pipeline.operators)

        if op isa StandardScaler
            !(op.params.with_mean || op.params.with_std) && begin
                op.scaler.mean_ = nothing
                op.scaler.var_  = nothing
                op.scaler.scale_ = nothing
                
                _act[i] = false; continue
            end
        end

        (!hasfield(typeof(op), :categories) || isa(op.categories, String)) && continue

        enc = op.encoder; enc.categories_ = enc.categories
        enc.n_features_in_ = length(op.categories); enc._infrequent_enabled = false

        if op isa OrdinalEncoder
            enc.feature_names_in_ = features[sortperm(reader.feature_idxs)]

        elseif op isa OneHotEncoder
            enc.feature_names_in_ = op.features
            enc._missing_indices = Dict{Int, Int}()
            _set_internal_onehot_state!(enc)
        end

        _act[i] = false
    end

    _act

end

print_stats(pipeline::Pipeline) = foreach(op -> (println("Operator: $(typeof(op))"); print_stats(op); println()), pipeline.operators)

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]