module Mneme

import ..torcjulia

include("offsets.jl")

include("./preprocessors/scalers/minmaxscaler.jl")
include("./preprocessors/scalers/maxabsscaler.jl")
include("./preprocessors/scalers/standardscaler.jl")

include("./preprocessors/encoders/ordinalencoder.jl")

export BlockReader, MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, fit, transform, print_stats

using .offsets

using .minmaxscaler
using .maxabsscaler
using .standardscaler

using .ordinalencoder

const BlockReader = offsets.BlockReader

const MinMaxScaler = minmaxscaler.MinMaxScaler
const MaxAbsScaler = maxabsscaler.MaxAbsScaler
const StandardScaler = standardscaler.StandardScaler

const OrdinalEncoder = ordinalencoder.OrdinalEncoder

fit(s::MinMaxScaler, reader) = minmaxscaler.fit(s, reader)
transform(s::MinMaxScaler, X) = minmaxscaler.transform(s, X)
print_stats(s::MinMaxScaler) = minmaxscaler.print_stats(s)

fit(s::MaxAbsScaler, reader) = maxabsscaler.fit(s, reader)
transform(s::MaxAbsScaler, X) = maxabsscaler.transform(s, X)
print_stats(s::MaxAbsScaler) = maxabsscaler.print_stats(s)

fit(s::StandardScaler, reader) = standardscaler.fit(s, reader)
transform(s::StandardScaler, X) = standardscaler.transform(s, X)
print_stats(s::StandardScaler) = standardscaler.print_stats(s)

fit(e::OrdinalEncoder, reader) = ordinalencoder.fit(e, reader)
transform(e::OrdinalEncoder, X) = ordinalencoder.transform(e, X)
print_stats(e::OrdinalEncoder) = ordinalencoder.print_stats(e)

function init(func::Function)
    torcjulia.start(func)       
end

end