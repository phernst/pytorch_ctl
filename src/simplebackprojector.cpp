#include "simplebackprojector.h"
#include "mat/projectionmatrix.h"
#include "simple_backprojector_kernel.h"
#include <iostream>

#ifdef OCL_CONFIG_MODULE_AVAILABLE
#include "ocl/openclconfig.h"
#endif

namespace CTL {

const std::string CL_KERNEL_NAME_BACKSMEAR = "simple_backprojector"; //!< name of the OpenCL kernel function
const std::string CL_KERNEL_NAME_NULLIFIER = "nullify_buffer"; //!< name of the OpenCL kernel function

SimpleBackprojector::SimpleBackprojector()
#ifdef OCL_CONFIG_MODULE_AVAILABLE
    : _ocl{new OpenCLHandler(this)}
{
    _ocl->initialize();
    _gpuMode = true;
}
#else
{}
#endif

#ifdef OCL_CONFIG_MODULE_AVAILABLE
SimpleBackprojector::SimpleBackprojector(uint oclDeviceNb)
    : _ocl{new OpenCLHandler(this, oclDeviceNb)}
{
    _ocl->initialize();
    _gpuMode = true;
}
#endif

VoxelVolume<float> SimpleBackprojector::backproject(const ProjectionData& projections)
{
    _volume = VoxelVolume<float>(_nbVoxels.get<0>(), _nbVoxels.get<1>(), _nbVoxels.get<2>(),
                                 _voxelSize.get<0>(), _voxelSize.get<1>(), _voxelSize.get<2>());
    _volume.setVolumeOffset(_offset.get<0>(), _offset.get<1>(), _offset.get<2>());
    _volume.fill(0.0f);

    if(!_gpuMode) // CPU version
    {
        auto pMats = _setup;

        // loop over modules
        const uint nbViews = projections.nbViews();
        const uint nbModules = projections.viewDimensions().nbModules;
        for(uint view = 0; view < nbViews; ++view)
        {
            for(uint mod = 0; mod < nbModules; ++mod)
            {
                processModule(projections.view(view).module(mod), pMats.get(view, mod));
            }
        }
    }
    else // GPU version
    {
#ifdef OCL_CONFIG_MODULE_AVAILABLE
        backprojectGPU(projections);
#endif
    }


    return std::move(_volume);
}

void SimpleBackprojector::configure(const ParallelSetup& setup)
{
    _setup = setup;
}


void SimpleBackprojector::setNbVoxels(uint x, uint y, uint z) { _nbVoxels = { double(x), double(y), double(z) }; }

void SimpleBackprojector::setVoxelSize(float xSize, float ySize, float zSize)
{
    _voxelSize = { xSize, ySize, zSize };
}

void SimpleBackprojector::setVolumeSpecs(const VoxelVolume<float>::Dimensions& nbVoxels,
                                         const VoxelVolume<float>::VoxelSize& voxelSize,
                                         const VoxelVolume<float>::Offset& offset)
{
    _nbVoxels = { double(nbVoxels.x), double(nbVoxels.y), double(nbVoxels.z) };
    _voxelSize = { voxelSize.x, voxelSize.y, voxelSize.z };
    _offset = { offset.x, offset.y, offset.z };
}

void SimpleBackprojector::setInterpolate(bool interpolate) { _interpolate = interpolate; }

void SimpleBackprojector::setVolumeSubsamplingFactor(uint volumeSubsamplingFactor)
{
    _volumeSubsamplingFactor = volumeSubsamplingFactor;
}

void SimpleBackprojector::processModule(const Chunk2D<float>& moduleData, const mat::ProjectionMatrix& P1)
{
    uint X = _nbVoxels.get<0>(), Y = _nbVoxels.get<1>(), Z = _nbVoxels.get<2>(); // abbreviation

    // center of the voxel that is at the corner of the volume
    Vector3x1 volCorner{ -0.5 * _voxelSize.get<0>() * double(X - 1),
                         -0.5 * _voxelSize.get<1>() * double(Y - 1),
                         -0.5 * _voxelSize.get<2>() * double(Z - 1) };
    volCorner += _offset;
    // upper bound for projection pixel index (nbPixels - 1 + 0.5, this bound is because
    // integer index is considered to be the center of a pixel)
    const double upperBoundX = double(moduleData.width()) - 0.5;
    const double upperBoundY = double(moduleData.height()) - 0.5;

    // temp variables within the following loop
    Vector3x1 p1_homo, p1_homoX, p1_homoY;
    mat::Matrix<2, 1> p1;
    // 3d world coord of voxel center (homog. form)
    mat::Matrix<4, 1> r({ 0.0, 0.0, 0.0, 1.0 });
    // the projection matrices splitted into columns
    const Vector3x1 P1Column0 = P1.column<0>();
    const Vector3x1 P1Column1 = P1.column<1>();
    const Vector3x1 P1Column2 = P1.column<2>();
    const Vector3x1 P1Column3 = P1.column<3>();

    // __Iteration over voxels__
    // use separation of matrix product:
    // (P*r)(i) = P(i,0)*r(0) + P(i,1)*r(1) + P(i,2)*r(2) + P(i,3)*r(3)
    for(uint x = 0; x < X; ++x)
    {
        r.get<0>() = std::fma(double(x), _voxelSize.get<0>(), volCorner.get<0>());
        // P*r = P(.,0)*r(0) + P(.,3)*r(3) + ...
        p1_homoX = P1Column0 * r.get<0>() + P1Column3; // * r(3)=1.0

        for(uint y = 0; y < Y; ++y)
        {
            r.get<1>() = std::fma(double(y), _voxelSize.get<1>(), volCorner.get<1>());
            // ... + P(.,1)*r(1)
            p1_homoY = p1_homoX + P1Column1 * r.get<1>();

            for(uint z = 0; z < Z; ++z)
            {
                r.get<2>() = std::fma(double(z), _voxelSize.get<2>(), volCorner.get<2>());
                // ... + P(.,2)*r(2)
                p1_homo = p1_homoY + P1Column2 * r.get<2>();

                // convert to cartesian coord (divide by w, where p_homo = [x y w])
                p1 = { { p1_homo.get<0>(), p1_homo.get<1>() } };
                p1 /= p1_homo.get<2>();

                // check if p1 is outside the detector
                if(p1.get<0>() < -0.5 || p1.get<0>() >= upperBoundX ||
                   p1.get<1>() < -0.5 || p1.get<1>() >= upperBoundY)
                    continue;

                float backsmearValue = _interpolate ? interpolateAt(moduleData, p1)
                                                    : moduleData(uint(p1.get<0>() + 0.499),
                                                                 uint(p1.get<1>() + 0.499));

                _volume(x, y, z) += backsmearValue;
            }
        }
    }
}

float SimpleBackprojector::interpolateAt(const Chunk2D<float>& moduleData, const mat::Matrix<2,1>& pos)
{
    return moduleData(uint(pos.get<0>() + 0.5), uint(pos.get<1>() + 0.5));
}

#ifdef OCL_CONFIG_MODULE_AVAILABLE
void SimpleBackprojector::enforceCPUMode(bool enabled)
{
    _gpuMode = !enabled;
}

void SimpleBackprojector::backprojectGPU(const ProjectionData &projections)
{
    auto pMats = _setup;
    std::vector<std::vector<float>> concPMats;
    std::vector<float> srcPositions;
    for(uint view = 0; view < projections.nbViews(); ++view)
    {
        concPMats.push_back(pMats.concatenatedStdVector(view));

        /*auto srcPos = pMats.view(view).module(0).sourcePosition();
        srcPositions.push_back(float(srcPos.get<0>()));
        srcPositions.push_back(float(srcPos.get<1>()));
        srcPositions.push_back(float(srcPos.get<2>()));*/
        srcPositions.push_back(0.f);
        srcPositions.push_back(0.f);
        srcPositions.push_back(0.f);
    }

    try // exception handling
    {
        _ocl->createCommandQueue();
        _ocl->setDimensions(projections.dimensions());

        // create buffers and pinned memory
        _ocl->createBuffers();
        _ocl->createPinnedMemory();

        // set kernel arguments
        _ocl->setFixedKernelArgs();

        // write buffers
        _ocl->writeFixedBuffers();

        uint transferredViews = 0;
        if(_ocl->nbBlocks() == 1)
        {
            // transfer all projections and source positions to device
            _ocl->transferSourcePositions(srcPositions.data(), srcPositions.size());
            _ocl->transferProjectionData(projections, 0, _setup.nbViews());
            transferredViews = _setup.nbViews();
        }

        const uint nbVoxelInSlice = _volume.nbVoxels().x * _volume.nbVoxels().y;
        for(uint z = 0; z < _volume.nbVoxels().z; ++z)
        {
            // set slice buffer to zero
            _ocl->startNullifierKernel();

            // smear back all views
            for(uint view = 0; view < projections.nbViews(); ++view)
            {
                // transfer projection matrices (all modules) for current view
                _ocl->transferProjectionMatrices(concPMats[view].data());

                // transfer next block of projections and source positions if required
                if(view >= transferredViews)
                {
                    _ocl->transferSourcePositions(srcPositions.data() + 3 * transferredViews, _ocl->viewsPerBlock());
                    _ocl->transferProjectionData(projections, transferredViews, _ocl->viewsPerBlock());

                    transferredViews += _ocl->viewsPerBlock();
                }

                _ocl->startBacksmearKernel(z, view);
            }

            // read result of slice into full volume
            _ocl->readoutResult(_volume.rawData() + z * nbVoxelInSlice);

            if(_ocl->nbBlocks() > 1)
                transferredViews = 0;
        }

    } catch(const cl::Error& err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")";
    } catch(const std::bad_alloc& except)
    {
        std::cerr << "Allocation error: " << except.what();
    } catch(const std::exception& except)
    {
        std::cerr << "std exception: " << except.what();
    }
}

SimpleBackprojector::OpenCLHandler::OpenCLHandler(SimpleBackprojector *parent, uint oclDeviceNb)
    : _parent(parent), _oclDeviceNb(oclDeviceNb)
{
    _zeroVec[0] = 0;
    _zeroVec[1] = 0;
    _zeroVec[2] = 0;
}

void SimpleBackprojector::OpenCLHandler::createCommandQueue()
{
    auto& oclConfig = OCL::OpenCLConfig::instance();
    if(!oclConfig.isValid())
        throw std::runtime_error("OpenCLConfig has not been initiated");

    // context and number of used devices
    auto& context = oclConfig.context();

    // command queue
    _queue = cl::CommandQueue(context, OCL::OpenCLConfig::instance().devices()[_oclDeviceNb]);
}

void SimpleBackprojector::OpenCLHandler::createBuffers()
{
    // context and number of used devices
    auto& context = OCL::OpenCLConfig::instance().context();

    const size_t nbVoxelInSlice = _parent->_volume.nbVoxels().x * _parent->_volume.nbVoxels().y;

    _volCornerBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3);
    _voxSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3);
    _srcPosBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * _viewsPerBlock);
    _sliceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * nbVoxelInSlice);
}

void SimpleBackprojector::OpenCLHandler::createPinnedMemory()
{
    // pinned memory objects
    auto& context = OCL::OpenCLConfig::instance().context();
    cl::Buffer pMatBufferPinned(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, _bytesForPMatBuffer);
    _pMatBufferDevice = cl::Buffer(context, CL_MEM_READ_ONLY, _bytesForPMatBuffer);

    cl::Image3D projMemPinned(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              cl::ImageFormat(CL_INTENSITY, CL_FLOAT),
                              _projBufferDim[0], _projBufferDim[1], _projBufferDim[2]);
    _projMemDevice = cl::Image3D(context, CL_MEM_READ_ONLY,
                                 cl::ImageFormat(CL_INTENSITY, CL_FLOAT),
                                 _projBufferDim[0], _projBufferDim[1], _projBufferDim[2]);

    // map pinned memory
    size_t row_pitch, slice_pitch;
    _projMemHost = (float*)_queue.enqueueMapImage(projMemPinned, CL_TRUE, CL_MAP_WRITE,
                                                  _zeroVec, _projBufferDim, &row_pitch, &slice_pitch);
    _pMatBufferHost = (float*)_queue.enqueueMapBuffer(pMatBufferPinned, CL_TRUE, CL_MAP_WRITE,
                                                      0, _bytesForPMatBuffer);
}

void SimpleBackprojector::OpenCLHandler::initialize()
{
    try // OCL exception catching
    {
        auto& oclConfig = OCL::OpenCLConfig::instance();
        // general checks
        if(!oclConfig.isValid())
            throw std::runtime_error("OpenCLConfig is not valid");

        // check if required kernels are already provided
        if(!oclConfig.kernelExists(CL_KERNEL_NAME_BACKSMEAR) ||
                !oclConfig.kernelExists(CL_KERNEL_NAME_NULLIFIER) )
        {
            const auto clSourceCode = getSimpleBackprojectorKernel();

            // add kernel to OCLConfig
            oclConfig.addKernel(CL_KERNEL_NAME_BACKSMEAR, clSourceCode);
            oclConfig.addKernel(CL_KERNEL_NAME_NULLIFIER, clSourceCode);
        }

        // Create kernel
        _kernelBacksmear = oclConfig.kernel(CL_KERNEL_NAME_BACKSMEAR);
        if(_kernelBacksmear == nullptr)
            throw std::runtime_error("kernel pointer (backsmearing) not valid");
        _kernelNullifier = oclConfig.kernel(CL_KERNEL_NAME_NULLIFIER);
        if(_kernelNullifier == nullptr)
            throw std::runtime_error("kernel pointer (nullifier) not valid");

    } catch(const cl::Error& err)
    {
        std::cerr << "OpenCL error: " << err.what() << "(" << err.err() << ")";
        throw std::runtime_error("OpenCL error");
    }
}

void SimpleBackprojector::OpenCLHandler::readoutResult(void *dest)
{
    const size_t nbVoxelInSlice = _parent->_volume.nbVoxels().x * _parent->_volume.nbVoxels().y;
    _queue.enqueueReadBuffer(_sliceBuffer, CL_TRUE, 0, sizeof(float) * nbVoxelInSlice, dest);
}

void SimpleBackprojector::OpenCLHandler::setDimensions(const ProjectionData::Dimensions& projDim)
{
    _projDim = projDim;

    _projBufferDim[0] = _projDim.nbChannels;
    _projBufferDim[1] = _projDim.nbRows;
    updateViewsPerBlock();

    _bytesForPMatBuffer = sizeof(float) * 12 * _projDim.nbModules;
}

void SimpleBackprojector::OpenCLHandler::setFixedKernelArgs()
{
    /*
     * __kernel void simple_backprojector
     * ( uint z,                              --> arg 0
     *   uint viewNb,                         --> arg 1
     *   uint nbModules,                      --> arg 2
     *   uint subSamplingFactor,              --> arg 3
     *   __constant float3* volCorner_mm,     --> arg 4
     *   __constant float3* voxelSize_mm,     --> arg 5
     *   __constant float* srcPosAllViews_mm, --> arg 6
     *   const __global float4* pMatsGlobal,  --> arg 7
     *   __local float4* pMatsLoc             --> arg 8
     *   __global float* sliceBuf,            --> arg 9
     *   __read_only image3d_t proj )         --> arg 10
     */

    _kernelNullifier->setArg(0, _sliceBuffer);

    _kernelBacksmear->setArg(2,  _projDim.nbModules);
    _kernelBacksmear->setArg(3,  _parent->_volumeSubsamplingFactor);
    _kernelBacksmear->setArg(4,  _volCornerBuffer);
    _kernelBacksmear->setArg(5,  _voxSizeBuffer);
    _kernelBacksmear->setArg(6,  _srcPosBuffer);
    _kernelBacksmear->setArg(7,  _pMatBufferDevice);
    _kernelBacksmear->setArg(8,  _bytesForPMatBuffer, nullptr);
    _kernelBacksmear->setArg(9,  _sliceBuffer);
    _kernelBacksmear->setArg(10, _projMemDevice);
}

void SimpleBackprojector::OpenCLHandler::startBacksmearKernel(uint slice, uint view)
{
    _kernelBacksmear->setArg(0, slice);
    _kernelBacksmear->setArg(1, view % _viewsPerBlock);

    _queue.enqueueNDRangeKernel(*_kernelBacksmear, cl::NullRange,
                                cl::NDRange(_parent->_volume.nbVoxels().x, _parent->_volume.nbVoxels().y));
}

void SimpleBackprojector::OpenCLHandler::startNullifierKernel()
{
    _queue.enqueueNDRangeKernel(*_kernelNullifier, cl::NullRange,
                                cl::NDRange(_parent->_volume.nbVoxels().x * _parent->_volume.nbVoxels().y));
}

void SimpleBackprojector::OpenCLHandler::writeFixedBuffers()
{
    const size_t X = _parent->_volume.nbVoxels().x;
    const size_t Y = _parent->_volume.nbVoxels().y;
    const size_t Z = _parent->_volume.nbVoxels().z;

    float voxSize[3]   = { _parent->_volume.voxelSize().x,
                           _parent->_volume.voxelSize().y,
                           _parent->_volume.voxelSize().z };
    float volCorner[3] = { -0.5f * voxSize[0] * float(X - 1) + _parent->_volume.offset().x,
                           -0.5f * voxSize[1] * float(Y - 1) + _parent->_volume.offset().y,
                           -0.5f * voxSize[2] * float(Z - 1) + _parent->_volume.offset().z };

    _queue.enqueueWriteBuffer(_volCornerBuffer, false, 0, sizeof(float) * 3, volCorner);
    _queue.enqueueWriteBuffer(_voxSizeBuffer, false, 0, sizeof(float) * 3, voxSize);
}

void SimpleBackprojector::OpenCLHandler::transferSourcePositions(void *src, size_t nbElements)
{
    _queue.enqueueWriteBuffer(_srcPosBuffer, CL_TRUE, 0, sizeof(float) * nbElements, src);
}

void SimpleBackprojector::OpenCLHandler::transferProjectionData(const ProjectionData &projections, uint startView, uint nbViews)
{
    float* dest = _projMemHost;
    const uint nbModules = projections.viewDimensions().nbModules;
    const uint elementsPerModule = projections.viewDimensions().nbChannels * projections.viewDimensions().nbRows;

    uint endView = startView + nbViews;
    bool needZeroPadding = false;

    if(endView > projections.nbViews())
    {
        endView = projections.nbViews();
        needZeroPadding = true;
    }

    for(uint view = startView; view < endView; ++view)
    {
        for(uint module = 0; module < nbModules; ++module)
        {
            std::copy_n(projections.view(view).module(module).rawData(), elementsPerModule, dest);
            dest += elementsPerModule;
        }
    }

    if(needZeroPadding)
    {
        uint reqZeros = (startView + nbViews - endView) * nbModules * elementsPerModule;
        std::fill_n(dest, reqZeros, 0.0f);
    }

    _queue.enqueueWriteImage(_projMemDevice, CL_TRUE, _zeroVec, _projBufferDim, 0, 0, _projMemHost);
}

void SimpleBackprojector::OpenCLHandler::transferProjectionMatrices(float* src)
{
    std::copy_n(src, 12 * _projDim.nbModules, _pMatBufferHost);
    _queue.enqueueWriteBuffer(_pMatBufferDevice, CL_TRUE, 0, _bytesForPMatBuffer,
                              _pMatBufferHost);
}

size_t SimpleBackprojector::OpenCLHandler::nbBlocks() const { return _nbBlocks; }

uint SimpleBackprojector::OpenCLHandler::viewsPerBlock() const { return _viewsPerBlock; }

void SimpleBackprojector::OpenCLHandler::updateViewsPerBlock()
{
    const auto& device = OCL::OpenCLConfig::instance().devices()[_oclDeviceNb];
    auto maxMemAlloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    auto maxDim3 = device.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

    _nbBlocks = 1;
    _viewsPerBlock = _parent->_setup.nbViews();
    size_t reqMemoryForProj;

    while(true)
    {
        // recalculate buffer specs
        _projBufferDim[2] =  _projDim.nbModules * _viewsPerBlock;
        reqMemoryForProj = sizeof(float) * _projBufferDim[2];

        // check if buffer specs are compatible
        if(_projBufferDim[2] > maxDim3 || reqMemoryForProj > maxMemAlloc)
        {
            _viewsPerBlock = static_cast<uint>(std::ceil(_viewsPerBlock/2.0));
            ++_nbBlocks;

            if(_viewsPerBlock <= 1u) // prevent endless loop
                break; // projections will not fit in cl::Image3D (too many modules)!
        }
        else
            break;
    }
}

#endif

} // namespace CTL
