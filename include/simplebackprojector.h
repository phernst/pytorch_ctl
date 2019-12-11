#ifndef SIMPLEBACKPROJECTOR_H
#define SIMPLEBACKPROJECTOR_H

#include "abstractbackprojector.h"
#include "mat/matrix_types.h"
#include "parallelsetup.h"

#ifdef OCL_CONFIG_MODULE_AVAILABLE
#include "ocl/openclconfig.h"
#endif

namespace CTL {

namespace mat{
class ProjectionMatrix;
}

class SimpleBackprojector : public AbstractBackprojector
{
public:
    SimpleBackprojector();

#ifdef OCL_CONFIG_MODULE_AVAILABLE
    explicit SimpleBackprojector(uint oclDeviceNb);
#endif

    VoxelVolume<float> backproject(const ProjectionData& projections) override;
    void configure(const ParallelSetup& setup) override;

    void setNbVoxels(uint x, uint y, uint z);
    void setVoxelSize(float xSize, float ySize, float zSize);
    void setVolumeSpecs(const VoxelVolume<float>::Dimensions& nbVoxels,
                        const VoxelVolume<float>::VoxelSize& voxelSize,
                        const VoxelVolume<float>::Offset& offset = { 0.0f, 0.0f, 0.0f });
    void setInterpolate(bool interpolate);
    void setVolumeSubsamplingFactor(uint volumeSubsamplingFactor);

#ifdef OCL_CONFIG_MODULE_AVAILABLE
    void enforceCPUMode(bool enabled);
#endif

private:
    bool _gpuMode = false;
    bool _interpolate = false;
    uint _volumeSubsamplingFactor = 1;

    Vector3x1 _nbVoxels;
    Vector3x1 _voxelSize;
    Vector3x1 _offset;

    VoxelVolume<float> _volume = VoxelVolume<float>(0,0,0);
    ParallelSetup _setup;

    void processModule(const Chunk2D<float>& moduleData, const mat::ProjectionMatrix& P1);
    static float interpolateAt(const Chunk2D<float>& moduleData, const mat::Matrix<2,1>& pos);

#ifdef OCL_CONFIG_MODULE_AVAILABLE
    class OpenCLHandler;
    friend class OpenCLHandler;
    std::unique_ptr<OpenCLHandler> _ocl;

    void backprojectGPU(const ProjectionData& projections);
#endif
};

#ifdef OCL_CONFIG_MODULE_AVAILABLE
class SimpleBackprojector::OpenCLHandler
{
public:
    OpenCLHandler(SimpleBackprojector* parent, uint oclDeviceNb = 0);

    void createBuffers();
    void createCommandQueue();
    void createPinnedMemory();
    void initialize();
    void readoutResult(void* dest);
    void setDimensions(const ProjectionData::Dimensions& projDim);
    void setFixedKernelArgs();
    void startBacksmearKernel(uint slice, uint view);
    void startNullifierKernel();
    void transferProjectionData(const ProjectionData &projections, uint startView, uint nbViews);
    void transferProjectionMatrices(float* src);
    void transferSourcePositions(void* src, size_t nbElements);
    void writeFixedBuffers();

    size_t nbBlocks() const;
    uint viewsPerBlock() const;

private:
    // host
    SimpleBackprojector* _parent;

    cl::CommandQueue _queue;
    cl::Kernel* _kernelBacksmear;
    cl::Kernel* _kernelNullifier;

    // regular buffers
    cl::Buffer _volCornerBuffer;
    cl::Buffer _voxSizeBuffer;
    cl::Buffer _srcPosBuffer;
    cl::Buffer _sliceBuffer;
    // pinned memory
    cl::Buffer _pMatBufferDevice;
    cl::Image3D _projMemDevice;
    float* _projMemHost;
    float* _pMatBufferHost;

    // dimension info
    ProjectionData::Dimensions _projDim;
    cl::size_t<3> _zeroVec, _projBufferDim;
    size_t _bytesForPMatBuffer, _nbBlocks;
    uint _viewsPerBlock;
    uint _oclDeviceNb;

    void updateViewsPerBlock();
};
#endif

} // namespace CTL

#endif // SIMPLEBACKPROJECTOR_H
