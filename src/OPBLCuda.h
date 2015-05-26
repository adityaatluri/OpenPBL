#ifndef OPBLCUDA_H
#define OPBLCUDA_H

#include<iostream>
#include"OPBLCudaUtil.h"

#define PI 3.14

#define NUM_SAMPLES 1024
#define NUM_LOBES 1024
#define NUM_DIRECTIONS 1024

struct BSDFSamplesStruct{
	int numSamples;
	int numValid;
	int numSpecularBRDFs;
	float pdf[NUM_LOBES];
	float weight[NUM_LOBES];
	vec4 dir[NUM_LOBES];
	__device__ void sample(vec4 wi,
							int lobeSamples,
							BSDFSamplesStruct *bs,
							int roughness,
							float *rand1,
							float *rand2);
/*	
We dont need this as we already assigned the lobes and samples
to 1024
__device__ void getNumSpecSamples(BSDFSamplesStruct *);
*/
};

struct BSDFValueStruct{
	vec4 value[NUM_DIRECTIONS];
	float pdf[NUM_DIRECTIONS];
	int numSamples;
};

struct LightSamplingStruct{
	vec4 P[NUM_SAMPLES];
	vec4 Ln[NUM_SAMPLES];
	vec4 Cl[NUM_SAMPLES];
	float pdf[NUM_SAMPLES];
	int numValid;
	__device__ void sample(LightSamplingStruct*);
};

struct LightEmmisionStruct{
	vec4 P[NUM_SAMPLES];
	vec4 Cl[NUM_SAMPLES];
	float pdf[NUM_SAMPLES];
};

__device__ void BRDFSample(struct vec4, int, BSDFSamplesStruct*);

unsigned Integrate();

#endif OPBLCUDA_H
