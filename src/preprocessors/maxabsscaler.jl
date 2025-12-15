module maxabsscaler

import ..offsets: BlockReader

using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

sklearn = pyimport("sklearn.preprocessing")
np = pyimport("numpy") 

using DataFrames, CSV

import ..torcjulia

mutable struct MaxAbsScaler
    scaler::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}
end

MaxAbsScaler(file::String, features::Vector{Symbol}; copy = true) =
    MaxAbsScaler(
        sklearn.MaxAbsScaler(copy = copy),
        file,
        features,
        Int[],
    )

function fit(scaler::MaxAbsScaler, reader::BlockReader)
    block_size, offsets = reader.block_size, reader.block_offsets
    scaler.feature_idxs = _map_features(scaler.features, reader.feature_idxs_map)
    
    file = scaler.file; features = scaler.features
    args = (file, block_size, scaler.feature_idxs)
    
    _partial_res = torcjulia.map(
        _partial_fit,
        offsets;
        chunksize = 1,
        args = args
    )
    
    _set_attributes(scaler.scaler, _reduce(_partial_res), features, scaler.feature_idxs)
end

function _set_attributes(scaler::Py, stats::Tuple{Py, Int}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    scaler.max_abs_ = stats[1]; scaler.n_samples_seen_ = stats[2]
    scaler.scale_ = stats[1]
    scaler.n_features_in_ = length(stats[1])
    scaler.feature_names_in_ = features[sortperm(feature_idxs)]
end

function _partial_fit(offset::Int, file::String, block_size::Int, 
                      feat_mapping::Vector{Int})::Tuple{Vector{Float64}, Int}
    X = _fetch_chunk(offset, file, block_size, feat_mapping)
    
    n_samples = nrow(X)
    max_abs = [maximum(abs.(col)) for col in eachcol(X)]

    
    max_abs, n_samples
end

function _fetch_chunk(offset::Int, file::String, block_size::Int, feat_mapping::Vector{Int})::DataFrame
    open(file, "r") do io
        seek(io, offset)

        csvfile = CSV.File(
            io;
            header = false,
            limit = block_size,
            select = feat_mapping
        )

        DataFrame(csvfile)
    end
end

function _reduce(stats::Vector{Tuple{Vector{Float64}, Int}})::Tuple{Py, Int}
    maxabs = stats[1][1]; total_samples = stats[1][2]

    @inbounds for i in 2:length(stats)
        s = stats[i]
        @. maxabs = max(maxabs, s[1])
        total_samples += s[2]
    end

    np.array(maxabs), total_samples
end

function transform(scaler::MaxAbsScaler, X) 
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    scaler.scaler.transform(np.array(X))
end

function print_stats(scaler::MaxAbsScaler)
    println("scale_: $(scaler.scaler.scale_)\
      \nn_features_in_: $(scaler.scaler.n_features_in_)\
      \nn_samples_seen_: $(scaler.scaler.n_samples_seen_)\
      \nfeature_names_in_: $(scaler.scaler.feature_names_in_)")

end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_scaler(scaler::MaxAbsScaler)::Py = scaler.scaler

end                                                                      