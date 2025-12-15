module Mneme

import ..torcjulia

include("offsets.jl")
include("./preprocessors/minmaxscaler.jl")
include("./preprocessors/maxabsscaler.jl")

export BlockReader, MinMaxScaler, MaxAbsScaler, fit, transform, print_stats

using .offsets
using .minmaxscaler
using .maxabsscaler

const BlockReader = offsets.BlockReader

const MinMaxScaler = minmaxscaler.MinMaxScaler
const MaxAbsScaler = maxabsscaler.MaxAbsScaler

fit(s::MinMaxScaler, reader) = minmaxscaler.fit(s, reader)
transform(s::MinMaxScaler, X) = minmaxscaler.transform(s, X)
print_stats(s::MinMaxScaler) = minmaxscaler.print_stats(s)

fit(s::MaxAbsScaler, reader) = maxabsscaler.fit(s, reader)
transform(s::MaxAbsScaler, X) = maxabsscaler.transform(s, X)
print_stats(s::MaxAbsScaler) = maxabsscaler.print_stats(s)

function init(func::Function)
    torcjulia.start(func)       
end

end