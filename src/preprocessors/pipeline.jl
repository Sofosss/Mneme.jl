using DataFrames, CSV

import ..torcjulia

mutable struct Pipeline
    operators::Vector{Union{MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder}}
    file::String

end

Pipeline(operators::Vector{Union{MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder}}, file::String) =
    Pipeline(operators, file)


abstract type OpDisp end

struct StandardDisp <: OpDisp end
struct MaxAbsDisp <: OpDisp end
struct MinMaxDisp <: OpDisp end
struct OrdinalDisp <: OpDisp end
struct OneHotDisp <: OpDisp end
struct LabelDisp <: OpDisp end

function fit(pipeline::Pipeline, reader::BlockReader)
    block_size, offsets = reader.block_size, reader.block_offsets

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
    
    file = pipeline.file; _disp = _disp_ops(_act_ops)
    args = (_disp, file, block_size, _feat_mapping, _operators_mapping)
    
    _partial_res = torcjulia.map(
        _partial_fit,
        offsets;
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

function _partial_fit(offset::Int, operators_disp::Vector{OpDisp}, file::String, block_size::Int, 
                      feat_mapping::Vector{Int}, _operators_mapping::Vector{<:AbstractVector{<:Integer}})
    chunk = _fetch_chunk(offset, file, block_size, feat_mapping)
    
    [_process_chunk(op_disp, chunk, _operators_mapping[i]) for (i, op_disp) in enumerate(operators_disp)]

end

function _fetch_chunk(offset::Int, file::String, block_size::Int, feat_mapping::Vector{Int})::DataFrame
    open(file, "r") do io
        seek(io, offset)
        csvfile = CSV.File(io; header = false, limit = block_size, select = feat_mapping)
        DataFrame(csvfile)
    end

end

function _get_union_features(operators, columns)
    features = vcat([op.features for op in operators]...)
    unique(col for col in columns if col in features)

end

function _disp_ops(operators)
    map(op -> begin
        if op isa StandardScaler
            StandardDisp()
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
    if disp isa StandardDisp
        n_samples = nrow(X)
        _mean = mean.(eachcol(X[:, feature_idxs]))
        _var  = varm.(eachcol(X[:, feature_idxs]), _mean; corrected = false)
        return _mean, _var, n_samples
    
    elseif disp isa MaxAbsDisp
        n_samples = nrow(X)
        max_abs = [maximum(abs.(col)) for col in eachcol(X[:, feature_idxs])]
        return max_abs, n_samples
    
    elseif disp isa MinMaxDisp
        n_samples = nrow(X)
        data_min = minimum.(eachcol(X[:, feature_idxs]))
        data_max = maximum.(eachcol(X[:, feature_idxs]))   

        return data_min, data_max, n_samples

    elseif disp isa LabelDisp
        return unique(X[:, feature_idxs[1]])
    
    end

    return [unique(X[:, i]) for i in sort(feature_idxs)]

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

        enc = op.encoder
        enc.categories_ = enc.categories
        enc.n_features_in_ = length(op.categories)
        enc._infrequent_enabled = false

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