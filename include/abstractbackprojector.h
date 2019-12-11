#ifndef ABSTRACTBACKPROJECTOR_H
#define ABSTRACTBACKPROJECTOR_H

#include <img/projectiondata.h>
#include <img/voxelvolume.h>
#include <parallelsetup.h>

namespace CTL {

class ProjectionData;


class AbstractBackprojector
{
    public:virtual VoxelVolume<float> backproject(const ProjectionData& projections) = 0;
    public:virtual void configure(const ParallelSetup& setup) = 0;

public:
    virtual ~AbstractBackprojector() = default;
};


} // namespace CTL

#endif // ABSTRACTBACKPROJECTOR_H
