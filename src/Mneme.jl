module Mneme

import ..torcjulia

include("offsets.jl")

include("./preprocessors/scalers/minmaxscaler.jl")
include("./preprocessors/scalers/maxabsscaler.jl")
include("./preprocessors/scalers/standardscaler.jl")

include("./preprocessors/encoders/ordinalencoder.jl")
include("./preprocessors/encoders/labelencoder.jl")
include("./preprocessors/encoders/onehotencoder.jl")

export BlockReader, MinMaxScaler, MaxAbsScaler, StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder, fit, transform, print_stats, to_JuliaCSR

using .offsets

using .minmaxscaler
using .maxabsscaler
using .standardscaler

using .ordinalencoder
using .labelencoder
using .onehotencoder

const BlockReader = offsets.BlockReader

const MinMaxScaler = minmaxscaler.MinMaxScaler
const MaxAbsScaler = maxabsscaler.MaxAbsScaler
const StandardScaler = standardscaler.StandardScaler

const OrdinalEncoder = ordinalencoder.OrdinalEncoder
const LabelEncoder = labelencoder.LabelEncoder
const OneHotEncoder = onehotencoder.OneHotEncoder

using PythonCall

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

fit(e::LabelEncoder, reader) = labelencoder.fit(e, reader)
transform(e::LabelEncoder, X) = labelencoder.transform(e, X)
print_stats(e::LabelEncoder) = labelencoder.print_stats(e)

fit(e::OneHotEncoder, reader) = onehotencoder.fit(e, reader)
transform(e::OneHotEncoder, X) = onehotencoder.transform(e, X)
print_stats(e::OneHotEncoder) = onehotencoder.print_stats(e)
to_JuliaCSR(Y::Py) = onehotencoder.to_JuliaCSR(Y)

function init(func::Function)
    torcjulia.start(func)       
end

end