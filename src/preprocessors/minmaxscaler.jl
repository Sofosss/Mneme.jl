module minmaxscaler

import ..offsets: BlockReader

using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

sklearn = pyimport("sklearn.preprocessing")

using DataFrames
using CSV

import ..torcjulia

mutable struct MinMaxScaler
    params::NamedTuple{(:feature_range, :copy, :clip), Tuple{Tuple{Float64,Float64}, Bool, Bool}}
    scaler::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}
end

struct MinMaxStats
    data_min::Vector{Float64}
    data_max::Vector{Float64}
    n_samples::Int
end

MinMaxScaler(file::String, features::Vector{Symbol}; feature_range = (0.0, 1.0), copy = true, 
            clip = false) =
    MinMaxScaler(
        (feature_range = feature_range, copy = copy, clip = clip),
        sklearn.MinMaxScaler(feature_range = feature_range, copy = copy, clip = clip),
        file,
        features,
        Int[],
    )

function fit(scaler::MinMaxScaler, reader::BlockReader)
    block_size, offsets, columns = reader.block_size, reader.block_offsets, reader.columns
    scaler.feature_idxs = _map_features(scaler.features, reader.feature_idxs_map)
    
    file = scaler.file; features = scaler.features; kwargs = NamedTuple(scaler.params)
    args = (file, block_size, columns, features, scaler.feature_idxs)
    
    _partial_res = torcjulia.torc_map(
        _partial_fit,
        offsets;
        chunksize = 1,
        args = args,
        kwargs...
    )

    f_stats = reduce(_partial_res)

    scaler.scaler.data_min_ = f_stats.data_min
    scaler.scaler.data_max_ = f_stats.data_max
    scaler.scaler.n_samples_seen_ = f_stats.n_samples

    frange = scaler.scaler.feature_range
    scaler.scaler.data_range_ = f_stats.data_max .- f_stats.data_min
    scaler.scaler.scale_ = (frange[1] - frange[0]) ./ (f_stats.data_max .- f_stats.data_min)
    scaler.scaler.min_ = frange[1] .- f_stats.data_min .* (frange[1] - frange[0]) ./ (f_stats.data_max .- f_stats.data_min)
    scaler.scaler.n_features_in_ = length(scaler.features)
    scaler.scaler.feature_names_in_ = features[sortperm(scaler.feature_idxs)]

end

function _partial_fit(offset, file, block_size, columns, features, feat_mapping; kwargs...)
    X = _fetch_chunk(offset, file, block_size, feat_mapping)
    n_samples = size(X, 1)
    data_min = minimum(X, dims = 1)[:]
    data_max = maximum(X, dims = 1)[:]   

    MinMaxStats(data_min, data_max, n_samples)
end

function _fetch_chunk(offset, file, block_size, feat_mapping)
    open(file, "r") do io
        seek(io, offset)

        csvfile = CSV.File(
            io;
            header = false,
            limit = block_size,
            select = feat_mapping
        )

        data = DataFrame(csvfile)
        Matrix(Float64.(data))
    end
end

function reduce(stats::Vector{MinMaxStats})
    reduce_minmax_scalers(stats)
end

function reduce_minmax_scalers(stats::Vector{MinMaxStats})::MinMaxStats
    _stats = popfirst!(copy(stats)) 
    _min = _stats.data_min
    _max = _stats.data_max
    total_samples = _stats.n_samples

    for s in stats
        _min = min.(_min, s.data_min)
        _max = max.(_max, s.data_max)
        total_samples += s.n_samples
    end

    MinMaxStats(_min, _max, total_samples)
end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_scaler(scaler::MinMaxScaler) = scaler.scaler

end
                                                                                                          