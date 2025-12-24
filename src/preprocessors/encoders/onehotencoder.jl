module onehotencoder

import ..offsets: BlockReader
import ..torcjulia

using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

using DataFrames, CSV, SparseArrays

sklearn = pyimport("sklearn.preprocessing")
np = pyimport("numpy")

mutable struct OneHotEncoder
    categories::Union{Vector, String}
    drop::Union{String, Vector, Nothing}
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

OneHotEncoder(
    file::String,
    features::Vector{Symbol};
    categories::Union{Vector, String} = "auto",
    drop::Union{String, Vector, Nothing} = nothing,
    dtype::Union{Type, Nothing} = Float64,
    handle_unknown::String = "error",
    sparse_output::Bool = true
) =
begin
    py_dtype = _py_dtype(dtype)
    OneHotEncoder(
        categories,
        drop,
        sklearn.OneHotEncoder(
            categories = categories isa String ? categories : [np.array(cat_feat) for cat_feat in categories],
            drop = drop,
            dtype = py_dtype,
            handle_unknown = handle_unknown,
            sparse_output = sparse_output
        ),
        file,
        features,
        Int[],
    )
end

function fit(encoder::OneHotEncoder, reader::BlockReader)
    if !isa(encoder.categories, String)
        encoder.encoder.categories_ = encoder.encoder.categories
        encoder.encoder.n_features_in_ = length(encoder.categories)
        encoder.encoder.feature_names_in_ = encoder.features
        
        encoder.encoder._missing_indices = Dict{Int, Int}()
        encoder.encoder._infrequent_enabled = false

        _set_internal_onehot_state!(encoder.encoder)
        return
    end

    block_size, offsets = reader.block_size, reader.block_offsets
    encoder.feature_idxs = _map_features(encoder.features, reader.feature_idxs_map)

    file = encoder.file; features = encoder.features
    args = (file, block_size, encoder.feature_idxs)

    _partial_res = torcjulia.torc_map(
        _partial_fit,
        offsets;
        chunksize = 1,
        args = args
    )

    _set_attributes(encoder.encoder, _reduce(_partial_res), features, encoder.feature_idxs, encoder.drop)
end

function _set_attributes(encoder::Py, stats::Tuple{Py, Dict{Int, Int}}, features::Vector{Symbol}, feature_idxs::Vector{Int}, drop::Union{String, Vector, Nothing})
    encoder.categories_ = stats[1]
    encoder.n_features_in_ = length(features)
    encoder.feature_names_in_ = features
    
    encoder._missing_indices = stats[2] # need them if the user doesn't provide them
    encoder._infrequent_enabled = false
    
    _set_internal_onehot_state!(encoder)
end

function set_missing_indices(categories::Vector{<:Vector})
    missing_indices = Dict{Int, Int}()

    for (feature_idx, categories_for_idx) in enumerate(categories)
        if any(ismissing.(categories_for_idx))
            missing_indices[feature_idx - 1] = length(categories_for_idx) - 1
        end
    end

    missing_indices
end

function _partial_fit(offset::Int, file::String, block_size::Int, feat_mapping::Vector{Int})::Vector{Vector{Union{Missing, Any}}}
    X = _fetch_chunk(offset, file, block_size, feat_mapping)
    reorder_perm = sortperm(sortperm(feat_mapping))     # keep sorted, because if we have >1 feature, they must be in order
    classes = [unique(X[:, i]) for i in reorder_perm]
    
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

function _reduce(stats::Vector{Vector{Vector{Union{Missing, Any}}}})::Tuple{Py, Dict{Int, Int}}
    cats = stats[1]

    @inbounds for i in 2:length(stats)
        new_cats = stats[i]

        cats = [
            sort(union(new, last))
            for (last, new) in zip(cats, new_cats)
        ]
    end

    missing_indices = set_missing_indices(cats)
    pylist([np.array(cat_feat) for cat_feat in cats]), missing_indices
end

function transform(encoder::OneHotEncoder, X)
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    Y = encoder.encoder.transform(np.array(X))

    # dense = numpy array - Matrix{Float64}
    # sparse = scipy sparse matrix - SparseMatrixCSC
    if pyhasattr(Y, "tocsr") || pyhasattr(Y, "tocsc")
        if !pyhasattr(Y, "indptr")
            Y = Y.tocsr()
        end
        return _csr_to_csc(Y; Tv=Float64, Ti=Int)
    else
        return pyconvert(Matrix{Float64}, Y)
    end
end

function _set_internal_onehot_state!(encoder::Py)
    # categories lsit per feature
    n_features = pyconvert(Int, pylen(encoder.categories_))
    
    py_none = pybuiltins.None
    drop_param = encoder.drop
    
    if !pyis(drop_param, py_none)
        # it is already string, but we need to convert it to python string
        drop_str = try
            pyconvert(String, drop_param)
        catch
            nothing
        end
        
        # sklearn disables binary drop when handle_unknown != 'error'
        if drop_str == "if_binary"
            encoder.drop_idx_ = py_none

        # for every feature, drop category index 0
        elseif drop_str == "first"
            encoder.drop_idx_ = np.zeros(n_features, dtype=np.int64)
        else
            encoder.drop_idx_ = drop_param
        end
    else
        encoder.drop_idx_ = py_none
    end
    
    # infrequently category grouping, we need to set it
    encoder._drop_idx_after_grouping = encoder.drop_idx_
    
    encoder._default_to_infrequent_mappings = pylist([py_none for _ in 1:n_features])
    encoder._infrequent_indices = pydict()
    
    # Calculate _n_features_outs (output columns per feature)
    n_features_out = Int[]
    drop_enabled = !pyis(encoder.drop_idx_, py_none)
    
    for cat_array in encoder.categories_
        n_cats = pyconvert(Int, pylen(cat_array))
        value_to_push = drop_enabled ? (n_cats - 1) : n_cats
        push!(n_features_out, value_to_push)
    end
    
    encoder._n_features_outs = pylist(n_features_out)
end

# copied from chatgpt
function _csr_to_csc(csr::Py; Tv=Float64, Ti=Int)
    data    = pyconvert(Vector{Tv}, csr.data)
    indices = pyconvert(Vector{Ti}, csr.indices)   # column indices (0-based)
    indptr  = pyconvert(Vector{Ti}, csr.indptr)    # row pointer (0-based)
    shape   = pyconvert(Tuple{Ti,Ti}, csr.shape)

    m, n = shape
    nnz = length(data)

    I = Vector{Ti}(undef, nnz)
    J = Vector{Ti}(undef, nnz)
    V = data

    k = 1
    @inbounds for r in 1:m
        # csr rows are 0-based in indptr
        start_ = indptr[r] + 1
        stop_  = indptr[r+1]
        for p in start_:stop_
            I[k] = r
            J[k] = indices[p] + 1
            k += 1
        end
    end

    return sparse(I, J, V, m, n)
end

function print_stats(encoder::OneHotEncoder)
    println("categories_: $(encoder.encoder.categories_)")
    println("n_features_in_: $(encoder.encoder.n_features_in_)")
    println("feature_names_in_: $(encoder.encoder.feature_names_in_)")
    println("drop_idx_: $(encoder.encoder.drop_idx_)")
end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_encoder(encoder::OneHotEncoder)::Py = encoder.encoder

end