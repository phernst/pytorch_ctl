// interpolating sampler with `0` as boundary color
__constant sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

float geometryCorrection(float3 directionVec, float3 voxelSize_mm);

// the nullifier kernel
__kernel void nullify_buffer( __global float* sliceBuf )
{
    const int X = get_global_size(0);
    const size_t idx = get_global_id(0) + get_global_id(1)*X;

    sliceBuf[idx] = 0.0f;
}

// the backsmear kernel
__kernel void simple_backprojector( uint z,
                                    uint viewNb,
                                    uint nbModules,
                                    uint subSamplingFactor,
                                    __constant float3* volCorner_mm,
                                    __constant float3* voxelSize_mm,
                                    __constant float* srcPosAllViews_mm,
                                    const __global float4* pMatsGlobal,
                                    __local float4* pMatsLoc,
                                    __global float* sliceBuf,
                                    __read_only image3d_t proj)
{
    event_t cpyToLocalEvent = async_work_group_copy(pMatsLoc, pMatsGlobal, 3 * nbModules, 0);
    // get IDs
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    // get dimensions
    const int X = get_global_size(0);
    const int Y = get_global_size(1);
    const size_t bufIdx = x + y*X; // 1D lookup index for z-slice buffer (result)
    const float4 projDim = convert_float4(get_image_dim(proj));

    const float3 srcPos_mm = (float3)(srcPosAllViews_mm[viewNb],
                                      srcPosAllViews_mm[viewNb+1],
                                      srcPosAllViews_mm[viewNb+2]);
    const float4 subSampleStep_mm = (float4)((*voxelSize_mm), 0.0f) / (float)subSamplingFactor;

    // world coordinate vector of voxel (center) position for first subsample
    const float4 rCorner = (float4)( x * (*voxelSize_mm).x + (*volCorner_mm).x - (subSamplingFactor-1) * 0.5f * subSampleStep_mm.x,
                                     y * (*voxelSize_mm).y + (*volCorner_mm).y - (subSamplingFactor-1) * 0.5f * subSampleStep_mm.y,
                                     z * (*voxelSize_mm).z + (*volCorner_mm).z - (subSamplingFactor-1) * 0.5f * subSampleStep_mm.z,
                                     1.0f);

    uint xSub, ySub, zSub;
    float geomCorr, totalCorr = 0.0f;
    float4 pMatRow0, pMatRow1, pMatRow2, projVal, r, p = (float4)0.0f;

    wait_group_events(1, &cpyToLocalEvent);

    // loop over all detector sub-modules
    for(zSub = 0; zSub < subSamplingFactor; ++zSub)
        for(ySub = 0; ySub < subSamplingFactor; ++ySub)
            for(xSub = 0; xSub < subSamplingFactor; ++xSub)
            {
                // shift to center of subsampled voxel
                r = rCorner + (float4)(xSub, ySub, zSub, 0.0f) * subSampleStep_mm;

                for(uint module = 0; module < nbModules; ++module)
                {
                    pMatRow0 = pMatsLoc[module * 3 + 0];
                    pMatRow1 = pMatsLoc[module * 3 + 1];
                    pMatRow2 = pMatsLoc[module * 3 + 2];

                    // get geometry correction factor
                    geomCorr = geometryCorrection((float3)(-pMatRow0.y,pMatRow0.x,0.0f), *voxelSize_mm);

                    //p.z = dot(pMatRow2, r);

                    p.x = dot(pMatRow0, r);// / p.z;
                    if(p.x < -1.0f || p.x > projDim.x)
                        continue;

                    p.y = dot(pMatRow1, r);// / p.z;
                    if(p.y < -1.0f || p.y > projDim.y)
                        continue;

                    p.z = module + viewNb*nbModules; // module index

                    p += 0.5f;

                    projVal = read_imagef(proj, samp, p);

                    totalCorr += projVal.x * geomCorr;
                }

            }

    sliceBuf[bufIdx] += totalCorr / pown((float)subSamplingFactor, 3);
}

float geometryCorrection(float3 directionVec, float3 voxelSize_mm)
{
    directionVec /= fmax(fmax(fabs(directionVec.x), fabs(directionVec.y)), fabs(directionVec.z));

    return length(directionVec * voxelSize_mm);
}