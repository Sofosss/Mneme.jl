

using Serialization
using CSV

struct BlockReader
    filepath::String
    n_rows::Int
    n_cols::Int
    columns::Vector{Symbol}
    feature_idxs_map::Dict{Symbol, Int}
    target::Vector{Symbol}
    block_size::Int
    n_blocks::Int
    block_offsets::Vector{Int}

    function BlockReader(filepath::String; num_blocks::Int = 1, num_rows::Int = -1,
                        target::Vector{Symbol} = Symbol[], offsets_path::String = "")
        
        if !isfile(filepath)
            throw(ArgumentError("CSV file not found: $filepath"))
        end

        if num_rows <= 0 && num_rows != -1
            throw(ArgumentError("num_rows must be positive, got: $num_rows"))
        end

        if num_blocks <= 0
            throw(ArgumentError("num_blocks must be positive, got: $num_blocks"))
        end
        
        n_rows = num_rows == -1 ? inspect_rows(filepath) : num_rows; n_cols, columns, feat_map = inspect_cols(filepath)

        block_size = ceil(Int, n_rows / num_blocks); target = isempty(target) ? [columns[end]] : target

        offsets_path = (!isempty(offsets_path) && !endswith(offsets_path, ".dat")) ? offsets_path * ".dat" : offsets_path

        offsets = (!isempty(offsets_path) && isfile(offsets_path)) ? fetch_block_offsets(offsets_path) : create_block_offsets(filepath, n_rows, block_size)

        new(filepath, n_rows, n_cols, columns, feat_map, target, block_size, length(offsets), offsets)
    end

end

function create_block_offsets(path::String, nrows::Int, block_size::Int)
    offsets = Int[]
    try
        open(path, "r") do file
            readline(file); offset = position(file); row = 0
            for line in eachline(file)
                row += 1
                if (row - 1) % block_size == 0
                    push!(offsets, offset)
                end
                offset += sizeof(line) + 1 
                if row == nrows
                    break
                end
            end
        end
    catch e
        throw(ErrorException("Failed to create block offsets from '$path': $(string(e))"))
    end
    
    offsets
end

function fetch_block_offsets(path::String)
    if !endswith(path, ".dat") 
        path *= ".dat"
    end

    if !isabspath(path)
        path = joinpath(pwd(), path)
    end

    if !isfile(path)
        throw(ErrorException("Block offsets file not found: '$path'"))
    end

    try
        open(path, "r") do f
            return deserialize(f)
        end
    catch e
        throw(ErrorException("Failed to load offsets from '$path': $(string(e))"))
    end

end

function get_offsets(reader::BlockReader)
    reader.block_offsets
end

function get_num_blocks(reader::BlockReader)
    reader.n_blocks
end

function save_offsets(offsets::Vector{Int}; path::String = "offsets.dat")
    if !endswith(path, ".dat") 
        path *= ".dat"
    end

    if !isabspath(path)
        path = joinpath(pwd(), path)
    end
        
    dir = dirname(path)

    if !isdir(dir)
        throw(ErrorException("Failed to save offsets. The directory '$dir' does not exist."))
    end

    try
        open(path, "w") do f
            serialize(f, offsets)
        end
    catch e
        throw(ErrorException("Failed to save offsets to '$path': $(string(e))"))
    end

end

function inspect_cols(path::String)
    file = CSV.File(path; header = false)
    cols = Symbol.(first(file))
    feature_idxs_map = Dict(col => i for (i, col) in enumerate(cols))

    length(cols), cols, feature_idxs_map
end

function inspect_rows(path::String)
    try
        output = read(`wc -l $path`, String)
        n = parse(Int, split(output)[1])
        return n - 1 
    catch
        cnt = 0
        open(path, "r") do f
            bufsize = 1 << 16  
            buf = Vector{UInt8}(undef, bufsize)
            while !eof(f)
                nread = read!(f, buf)
                count += count(c -> c == UInt8('\n'), buf[1:nread])
            end
        end
        return cnt - 1
    end
end