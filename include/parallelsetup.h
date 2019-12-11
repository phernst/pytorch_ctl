#ifndef PARALLELSETUP_H
#define PARALLELSETUP_H

#include <vector>

#include "mat/projectionmatrix.h"
#include "img/voxelvolume.h"

class ParallelSetup
{
public:
    ParallelSetup() = default;
    ParallelSetup(const std::vector<float>& theta, float offet);
    const CTL::mat::ProjectionMatrix& get(uint view, uint module) const;
    uint nbViews() const { return _pMats.size(); }
    std::vector<float> concatenatedStdVector(uint view) const;

private:
    std::vector<std::vector<CTL::mat::ProjectionMatrix>> _pMats;
};

#endif