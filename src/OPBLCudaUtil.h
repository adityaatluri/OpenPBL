#ifndef OPBLCUDAUTIL_H
#define OPBLCUDAUTIL_H

#include<cuda.h>
#include"cuda_runtime.h"

struct vec4{
	union{
		struct{ float x, y, z, w; };
		struct{ float r, g, b, a; };
	};
};


__device__ struct vec4& operator+=(struct vec4& in1, struct vec4& in2){
	in1.x += in2.x;
	in1.y += in2.y;
	in1.z += in2.z;
	in1.w += in2.w;
	return in1;
}

__device__ bool& operator!=(struct vec4& in1, struct vec4& in2){
	bool st;
	if (in1.x == in2.x){
		st = true;
	}
	if (in1.y == in2.y){
		st = true;
	}
	if (in1.z == in2.z){
		st = true;
	}
	if (in1.w == in2.w){
		st = true;
	}
	return st;
}

__device__ struct vec4& operator/(struct vec4& in1, struct vec4& in2){
	struct vec4 tmp;
	tmp.x = in1.x / in2.x;
	tmp.y = in1.y / in2.y;
	tmp.z = in1.z / in2.z;
	tmp.w = in1.w / in2.w;
	return tmp;
}

__device__ struct vec4& operator/(struct vec4& in1, float& f){
	struct vec4 tmp;
	tmp.x = in1.x / f;
	tmp.y = in1.y / f;
	tmp.z = in1.z / f;
	return tmp;
}

__device__ struct vec4& operator+(struct vec4& in1, struct vec4& in2){
	struct vec4 tmp;
	tmp.x = in1.x + in2.x;
	tmp.y = in1.y + in2.y;
	tmp.z = in1.z + in2.z;
	tmp.w = in1.w + in2.w;
	return tmp;
}

__device__ struct vec4& operator*(struct vec4& in1, struct vec4& in2){
	struct vec4 tmp;
	tmp.x = in1.x * in2.x;
	tmp.y = in1.y * in2.y;
	tmp.z = in1.z * in2.z;
	tmp.w = in1.w * in2.w;
	return tmp;
}

__device__ float& dot(struct vec4& in1, struct vec4& in2){
	float tmp;
	tmp = in1.x * in2.x + in1.y * in2.y + in1.z * in2.z + in1.w * in2.w;
	return tmp;
}

__device__ struct vec4& operator*(float f, struct vec4& in){
	struct vec4 tmp;
	tmp.x = f * in.x;
	tmp.y = f * in.y;
	tmp.z = f * in.z;
	tmp.w = f * in.w;
	return tmp;
}

__device__ struct vec4& operator-(struct vec4& in1, struct vec4& in2){
	struct vec4 tmp;
	tmp.x = in1.x - in2.x;
	tmp.y = in1.y - in2.y;
	tmp.z = in1.z - in2.z;
	tmp.w = in1.w - in2.w;
	return tmp;
}

__device__ struct vec4& operator-(float& f, struct vec4& in){
	struct vec4 tmp;
	tmp.x = f - in.x;
	tmp.y = f - in.y;
	tmp.z = f - in.z;
	tmp.w = f - in.w;
	return tmp;
}


__device__ struct vec4& normalize(struct vec4& in){
	struct vec4 tmp, one = { 1.0f, 1.0f, 1.0f, 1.0f };
	tmp.x = in.x / sqrt(one*in);
	tmp.y = in.y / sqrt(one*in);
	tmp.z = in.z / sqrt(one*in);
	tmp.w = in.w / sqrt(one*in);
	return tmp;
}

#endif
