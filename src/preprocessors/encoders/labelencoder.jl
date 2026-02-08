
using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

sklearn = pyimport("sklearn.preprocessing")
np = pyimport("numpy") 

using DataFrames, CSV

import ..torcjulia

mutable struct LabelEncoder
    encoder::Py
    file::String
    features::Symbol
    feature_idxs::Vector{Int}

end

LabelEncoder(
    file::String,
    features::Symbol
) =
    begin
        LabelEncoder(
            sklearn.LabelEncoder(),
            file,
            features,
            Int[],
        )
    end

function fit(encoder::LabelEncoder, reader::BlockReader)

    block_size, offsets = reader.block_size, reader.block_offsets
    encoder.feature_idxs = _map_features([encoder.features], reader.feature_idxs_map)
    
    file = encoder.file
    args = (file, block_size, encoder.feature_idxs)
    
    _partial_res = torcjulia.map(
        _partial_fit_le,
        offsets;
        chunksize = 1,
        args = args
    )
    
    _set_attributes_le(encoder.encoder, _reduce_le(_partial_res))

end

function _set_attributes_le(encoder::Py, classes::Py)
    encoder.classes_ = classes

end

function _partial_fit_le(offset::Int, file::String, block_size::Int, 
                      feat_mapping::Vector{Int})::Vector{Union{Missing, Any}}
    X = _fetch_chunk(offset, file, block_size, feat_mapping)
    classes = unique(X[:, 1])
    
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

function _reduce_le(stats)::Py
    classes = stats[1]

    @inbounds for i in 2:length(stats)
        new_classes = stats[i]

        classes = sort(union(new_classes, classes))
    end

    np.array(classes)

end

function transform(encoder::LabelEncoder, X) 
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    encoder.encoder.transform(np.asarray(X).ravel())

end

function print_stats(encoder::LabelEncoder)
    println("classes_: $(encoder.encoder.classes_)")
    
end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_encoder(encoder::LabelEncoder)::Py = encoder.encoder
