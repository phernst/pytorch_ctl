#include "parallelsetup.h"

ParallelSetup::ParallelSetup(const std::vector<float>& theta, float offset) // angles in radiant
{
    for (const auto& th : theta)
    {
        _pMats.push_back({CTL::mat::ProjectionMatrix{
            std::cos(th), std::sin(th), 0, offset,
            0, 0, 0, 0,
            0, 0, 0, 0}});
    }
}

const CTL::mat::ProjectionMatrix& ParallelSetup::get(uint view, uint module) const
{
    return _pMats[view][module];
}

std::vector<float> ParallelSetup::concatenatedStdVector(uint view) const
{
    const auto& viewvec { _pMats[view] };
    std::vector<float> ret;

    for(int mod = 0; mod < viewvec.size(); ++mod) {
        const auto& modmat { viewvec.at(mod) };
        for(uint el = 0; el < 12u; ++el) {
            ret.push_back(static_cast<float>(modmat(el)));
        }
    }

    return ret;
}
