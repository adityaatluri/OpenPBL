#include"OPBLCuda.h"

using namespace opblcuda;

__device__ void sample(vec4 wi,
							int lobeSamples,
							BSDFSamplesStruct *bs,
							int roughness,
							float *rand1,
							float *rand2){
	float ratio = bs->numSamples / lobeSamples;
	int numCurrent = bs->numValid;
	float roughness2 = roughness*roughness;
	for (int i = 0; i < lobeSamples; i++){
		float tantheta2 = logf(rand1[i] * roughness2);
		float costheta = 1 / sqrt(1+tantheta2);
		float costheta2 = costheta * costheta;
		float costheta3 = costheta * costheta2;
		vec4 H = SpericalDir(sqrt(1-costheta2), costheta, 2*PI*rand2[i]);
		float VdotH = dot(wi,H);
		vec4 wo =  2.0f * VdotH * H - wi;
		if (dot(wo,H) > 0){
			bs->pdf[numCurrent] = rand1[i] * ratio / (4 * PI * costheta3 * roughness2 * abs(VdotH));
			bs->weight[numCurrent] = specColor / ratio;
			bs->dir[numCurrent] = wo;
			numCurrent++;
		}
	}
	bs->numValid = numCurrent;
}

/*
Algorithm 1
*/
__device__ void sample(LightSamplingStruct *ls,
						vec4 lightCenterPos,
						vec4 P, float radius,
						float rayWeight,
						float *rand1){
	vec4 lightCenterDir = lightCenterPos - P;
	float d2 = dot(lightCenterDir,lightCenterDir);
	float radius2 = radius*radius;
	if (d2 - radius2 > 1e-4){
		float d = sqrt(d2);
		vec4 ONBU, ONBV, ONBW;
		CreateBasisFromW(lightCenterDir/d, ONBU, ONBV, ONBW);
		float solidAngle = 1 - sqrt(d2 / (radius2 + d2));
		int numSamples = ceil(rayWeight * solidAngle * NUM_SAMPLES);
		ls->numValid = numSamples;
		float costhetamax = sqrt(1 - (radius2/d2));
		float pdf = 1 / (2 * PI*(1 - costhetamax));
		for (int i = 0; i < numSamples; i++){
			float costheta = 1 + rand1[i] * (costhetamax-1);
			float sin2theta = 1 - costheta * costheta;
			vec4 lightDir = TransformFromBasis(lightDir, ONBU, ONBV, ONBW);
			float delta = sqrt(radius2 - sin2theta*d2);
			ls->P[i] = P + (costheta * d - delta) * lightDir;
			ls->Ln[i] = lightDir;
			ls->pdf[i] = pdf;
			ls->Cl[i] = lightColor;
		}
	}
	else{
		ls->numValid = 0;
	}
}

/*
Algorithm 3
*/
__device__ void diffConvolution(vec4 *diffColor,
								float radius,
								vec4 lightColor,
								vec4 lightCenterPos,
								vec4 P,
								vec4 N){
	vec4 zero = { 0.0f, 0.0f, 0.0f, 0.0f };
	float diffConv = 0;
	if (lightColor != zero){
		float radius2 = radius * radius;
		vec4 lightCenterDir = lightCenterPos - P;
		float d2 = dot(lightCenterDir,lightCenterDir);
		float cosTheta = dot(lightCenterDir,N) / (sqrt(d2));
		if (d2 - radius2 > 1e-4){
			float d = sqrt(d2);
			float sinAlpha = radius / d;
			float sinAlpha2 = sinAlpha * sinAlpha;
			float cosAlpha = sqrt(1 - sinAlpha2);
			float alpha = asin(sinAlpha);
			float theta = acos(cosTheta);
			if (theta < (PI / 2 - alpha)){
				diffConv = cosTheta * sinAlpha2;
			}
			else{
				if (theta < PI/2){
					float g0 = sinAlpha2 * sinAlpha;
					float g1 = (alpha - cosAlpha * sinAlpha) / PI;
					float gp0 = -cosAlpha*sinAlpha2*alpha;
					float gp1 = -sinAlpha2*alpha / 2;
					float a = gp1 + gp0 - 2 * (g1 - g0);
					float b = 3 * (g1 - g0) - gp1 - 2 * gp0;
					float y = (theta - (PI / 2 - alpha)) / alpha;
					diffConv = g0 + y * (gp0 + y * (b + y*a));
				}
				else{
					if (theta < (PI/2 + alpha)){
						float g0 = sinAlpha2 * sinAlpha;
						float gp0 = -(sinAlpha2*alpha) / 2;
						float a = gp0 + 2 * g0;
						float b = -3 * g0 - 2 * gp0;
						float y = (theta - PI / 2) / alpha;
						diffConv = g0 + y * (gp0 + y*(b+y*a));
					}
				}
			}
		}
		diffColor[0] = diffConv * lightColor;
	}
}

/*
Algorithm 13
*/
__device__ void computeBRDFShadows(vec4* CspecBRDF, vec4 *SpecDiff){
	/*
	size of:
	LvisBRDF = numActiveSpecSamples
	CspecBRDF = numActiveSpecSamples
	*/
	vec4 *LvisBRDF;
	vec4 one = {1.0f, 1.0f, 1.0f, 1.0f};
	AreaShadowRays(lightValues->P, bs->pdf, LvisBRDF);
	for (int i = 0; i < numActiveSpecSamples; i++){
		SpecDiff[0] += CspecBRDF[i] * (one - LvisBRDF[i]);
		SpecDiff[1] += CspecBRDF[1];
	}
}

/*
Algorithm 12
*/
__device__ void computeLightShadows(LightSamplingStruct *ls,
									LightSamplingStruct *li,
									vec4 *Cdiff, vec4 *Cspec, vec4 *SpecDiff){
	/*
	SpecDiff[0] = diffPerLight
	SpecDiff[1] = diffPerLightNoShad
	SpecDiff[2] = specPerLight
	SpecDiff[3] = specPerLightNoShad
	*/
	vec4 *Lvis;
	vec4 one = { 1.0f, 1.0f, 1.0f, 1.0f };
	vec4 avgVis = AreaShadowRays(ls->P, ls->pdf, Lvis);
	for (int i = 0; i < numGeneratedLightSamples; i++){
		SpecDiff[0] += Cdiff[i] * (one - Lvis[i]);
		SpecDiff[1] += Cdiff[i];
	}
	if (facingRatio >= 0){
		for (unsigned i = 0; i < numGeneratedLightSamples; i++){
			SpecDiff[2] += Cspec[i] * (one - Lvis[i]);
			SpecDiff[3] += Cspec[i];
		}
	}
	vec4 diffConv;
	if (li->diffConvolution(diffConv) == true){
		diffConv *= bsdf->albedo();
		SpecDiff[0] += (one - avgVis) * (diffConv - SpecDiff[1]);
		SpecDiff[1] = diffConv;
	}
}

/*
Algorithm 11
ComputeMIS needs a loads of revision and thought!
*/
__device__ vec4* ComputeMIS(LightSamplingStruct *ls,
							BSDFValueStruct *diffValues,
							BSDFValueStruct *specValues,
							BRDFStruct bs,
							LightEmmisionStruct *lightValues){
	vec4 C[3];
	/*
	C[0] = Cdiff
	C[1] = Cspec
	C[2] = CspecBRDF
	*/
	if (diffResampPercentage >= 1.0f){
		integrateIS(diffValues, ls, C+0);
	}
	else{
		integrateRIS(diffResampPercentage, diffValues, ls, C+0);
	}
	if (facingRatio >= 0.0f){
		if (specResampPercentage >= 1.0f){
			integrateMIS(ls, specValues, numLobeSamples, C+1, bs, lightValues, C+2);
		}
		else{
			integrateRMIS(specResampPercentage, ls, specValues, numLobeSamples, C+1, bs, lightValues, C+2);
		}
	}
}

/*
Algorithm 2
*/
__device__ void emissionAndPDF(BSDFSamplesStruct *bs,
								LightEmmisionStruct *le,
								vec4 P, vec4 lightCenterPos,
								float radius,
								vec4 lightColor){
	vec4 lightCenterDir = P - lightCenterPos;
	vec4 zero = {0.0f, 0.0f, 0.0f, 1.0f};
	float d2 = dot(lightCenterDir, lightCenterDir);
	float radius2 = radius*radius;
	bool isValid = false;
	if (d2 - radius2 >= 1e-4){
		float costhetamax = sqrt(1 - radius2/d2);
		float pdf = 1 / (2*PI*(1-costhetamax));
		for (int i = 0; i < bs->numValid; i++){
			bool isValid = false;
			vec4 dir = bs->dir[i];
			float b = 2 * dot(dir,lightCenterDir);
			float c = dot(lightCenterDir , lightCenterDir) - radius2;
			float delta = b * b - 4 * c;
			if (delta > 0){
				float t = (-b - sqrt(delta)) / 2;
				if (t < 1e-5){
					t = (-b+sqrt(delta))/2;
				}
				if (t >= 1e-5 && t <= 1e20){
					isValid = true;
					le->P[i] = P + t*dir;
					le->Cl[i] = lightColor;
					le->pdf[i] = pdf;
				}
			}
			if (isValid == false){
				le->Cl[i] = zero;
				le->pdf[i] = 0.0f;
			}
		}
	}
}

/*
Algorithm 5
*/
__device__ void valueAndPDF(vec4 wi, vec4* wos,
							BSDFValueStruct *bv,
							vec4 N, vec4 specColor,
							float roughness){
	vec4 wo;
	vec4 zero = { 0.0f, 0.0f, 0.0f, 1.0f };
	for (int i = 0; i < bv->numSamples; i++){
		wo = wos[i];
		if (dot(wo,N) > 0){
			hasValidValues = true;
			vec4 H = normalize(wo+wi);
			float costheta = dot(H,N);
			float costheta2 = costheta*costheta;
			float costheta3 = costheta2* costheta;
			float roughness2 = roughness*roughness;
			float pdf = exp((costheta2 - 1)/(roughness2 * costheta2))/(4*PI*costheta3*roughness2*dot(wi,H));
			bv->value[i] = pdf*specColor;
			bv->pdf[i] = pdf;
		}
		else{
			bv->value[i] = zero;
			bv->pdf[i] = 0.0f;
		}
	}
}

/*
Algorithm 7
It is parallelized here by assigning each thread to a specular BRDF

wouts length is NUM_DIRECTIONS
According to this function, the number of directions is equal to
number of samples.
*/
__device__ void valueAndPDF_Spec(
	BSDFValueStruct *bv,
	BSDFValueStruct *onebv,
	vec4 wi, vec4* wouts){
	valueAndPDF(wi, wouts, onebv);
	vec4 zero = { 0.0f, 0.0f, 0.0f, 1.0f };
	for (int k = 0; k < NUM_DIRECTIONS; k++){
		onebv->pdf[k] = 0.0f;
		onebv->value[k] = zero;
	}
	// TODO
	push(bv, onebv);
}

/*
Algorithm 10
*/
__global__ void IntegrateBRDF(
	BSDFSamplesStruct *bs,
	BSDFSamplesStruct *bsbvh,
	vec4 *wi,
	vec4 P,
	int numSpecLobes,
	int numSpecSamples,
	float roughness,
	float **rand1,
	float **rand2){

	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	bs->sample(wi[tid], numSpecSamples, &bs[tid], roughness, rand1[tid], rand2[tid]);
	/* TODO
	    |
	   \|/
	*/
	bsbvh = BVHReduce(P, bs);
}

/*
Algorithm 9
*/
__global__ void IntegrateLight(
	LightSamplingStruct *ls,
	BSDFSamplesStruct *bs,
	BSDFSamplesStruct *bsdf,
	BSDFValueStruct *bv,
	BSDFValueStruct *onebv,
	vec4 lightCenterPos, vec4 *P,
	float radius, float *rayWeight,
	float **rand1){

	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	ls[tid].sample(&ls[tid], lightCenterPos, P[tid], radius, rayWeight[tid], rand1[tid]);
	int numGenerateLightSamples = ls[tid].numValid;
	int numActiveSpecSamples = bs[tid].numValid;
	if (numGenerateLightSamples > 0){
		valueAndPDF_Diff(diffValues);
		valueAndPDF_Spec(specValues);
	}
	emissionAndPDF(bs, lightValues);
	ComputeMIS(ls, diffValues, specValues, bs, lightValues, Cdiff, Cspec, CspecBRDF);
	if (numGenerateLightSamples > 0){
		computeLightShadows(ls, li, Cdiff, Cspec, C);
	}
	if (numActiveSpecSamples > 0){
		computeBRDFShadows(CspecBRDF, specPerLight, specPerLightNoShad);
	}
	lightDiff = mix(C[1], C[0], shadowDensity);
	lightSpec = mix(C[3], C[2], shadowDensity);
}


/*
Algorithm 6
*/
unsigned Integrate(){
// Call IntegrateBrdf cuda code here!
// Call IntegrateLight cuda code here!
}
