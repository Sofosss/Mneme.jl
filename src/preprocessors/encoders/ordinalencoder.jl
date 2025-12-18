module ordinalencoder

import ..offsets: BlockReader

using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

sklearn = pyimport("sklearn.preprocessing")
np = pyimport("numpy") 

using DataFrames, CSV

import ..torcjulia

mutable struct OrdinalEncoder
    categories::Union{Vector{Union{String, Number}}, String}
    encoder::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}
end

_py_dtype(::Type{Float64}) = np.float64
_py_dtype(::Type{Float32}) = np.float32
_py_dtype(::Type{Int64})   = np.int64
_py_dtype(::Type{Int32})   = np.int32
_py_dtype(::Nothing)       = nothing

OrdinalEncoder(
    file::String,
    features::Vector{Symbol};
    categories::Union{Vector{Union{String, Number}}, String} = "auto", 
    dtype::Union{Type, Nothing} = Float64,
    handle_unknown::String = "error",
    unknown_value::Union{Int, Nothing} = nothing, 
    encoded_missing_value::Union{Int, String, Nothing} = nothing
) =
    begin
        py_dtype = _py_dtype(dtype) 
        OrdinalEncoder(
            categories,
            sklearn.OrdinalEncoder(
                categories = [np.array(cat_feat) for cat_feat in categories],
                dtype = py_dtype,
                handle_unknown = handle_unknown,
                unknown_value = unknown_value,
                encoded_missing_value = encoded_missing_value
            ),
            file,
            features,
            Int[],
        )
    end

function fit(encoder::OrdinalEncoder, reader::BlockReader)
    if !isa(encoder.categories, String)
        encoder.encoder.categories_ = encoder.encoder.categories 
        encoder.encoder.n_features_in_ = length(encoder.categories)
        encoder.encoder.feature_names_in_ = features[sortperm(reader.feature_idxs)]
        encoder.encoder._infrequent_enabled = false
        
        return
    end

    block_size, offsets = reader.block_size, reader.block_offsets
    encoder.feature_idxs = _map_features(encoder.features, reader.feature_idxs_map)
    
    file = encoder.file; features = encoder.features
    args = (file, block_size, encoder.feature_idxs)
    
    _partial_res = torcjulia.map(
        _partial_fit,
        offsets;
        chunksize = 1,
        args = args
    )
    
    _set_attributes(encoder.encoder, _reduce(_partial_res), features, encoder.feature_idxs)
end

function _set_attributes(encoder::Py, stats::Tuple{Vector{Py}, Dict{Int, Int}}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    encoder.categories_ = stats[1]
    encoder.n_features_in_ = length(features)
    encoder.feature_names_in_ = features[sortperm(feature_idxs)]
    
    encoder._missing_indices = stats[2]
    encoder._infrequent_enabled = false

end

function set_missing_indices(categories::Vector{<:Vector})
    missing_indices = Dict{Int, Int}()

    for (feature_idx, categories_for_idx) in enumerate(categories)
        missing_indices[feature_idx - 1] = length(categories_for_idx) - 1
    end

    missing_indices
end

function _partial_fit(offset::Int, file::String, block_size::Int, 
                      feat_mapping::Vector{Int})::Vector{Vector{Union{Missing, Any}}}
    X = _fetch_chunk(offset, file, block_size, feat_mapping)
    classes = [unique(col) for col in eachcol(X)]
    
    classes
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

function _reduce(stats::Vector{Vector{Vector{Union{Missing, Any}}}})::Tuple{Vector{Py}, Dict{Int, Int}}
    cats = stats[1]

    @inbounds for i in 2:length(stats)
        new_cats = stats[i]

        cats = [
            sort(union(new, last))
            for (last, new) in zip(cats, new_cats)
        ]
    end

    missing_indices = set_missing_indices(cats)

    [np.array(cat_feat) for cat_feat in cats], missing_indices
end

function transform(encoder::OrdinalEncoder, X) 
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    encoder.encoder.transform(np.array(X))
end

function print_stats(encoder::OrdinalEncoder)
    println("categories_: $(encoder.encoder.categories_)\
      \nn_features_in_: $(encoder.encoder.n_features_in_)\
      \nfeature_names_in_: $(encoder.encoder.feature_names_in_)")
end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_encoder(encoder::OrdinalEncoder)::Py = encoder.encoder

end 