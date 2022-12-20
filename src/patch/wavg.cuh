#ifndef CUDA_WAVG_KERNEL_CUH_
#define CUDA_WAVG_KERNEL_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"

template<bool REFCTF, bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_wavg(
		XFLOAT *g_eulers,
		AccProjectorKernel projector,
		unsigned image_size,
		unsigned long orientation_num,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		XFLOAT *g_wdiff2s_AA,
		XFLOAT *g_wdiff2s_XA,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale)
{
	XFLOAT ref_real, ref_imag, img_real, img_imag, trans_real, trans_imag;

	int bid = blockIdx.x; //block ID
	int tid = threadIdx.x;

	extern __shared__ XFLOAT buffer[];

	unsigned pass_num(ceilfracf(image_size,block_sz)),pixel;
	XFLOAT * s_wdiff2s_parts	= &buffer[0];
	XFLOAT * s_sumXA			= &buffer[block_sz];
	XFLOAT * s_sumA2			= &buffer[2*block_sz];
	XFLOAT * s_eulers           = &buffer[3*block_sz];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];
	__syncthreads();

	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel<image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					s_eulers[0], s_eulers[1], s_eulers[2],
					s_eulers[3], s_eulers[4], s_eulers[5],
					s_eulers[6], s_eulers[7], s_eulers[8],
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					s_eulers[6], s_eulers[7],
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
						x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					ref_real, ref_imag);

			if (REFCTF)
			{
				ref_real *= __ldg(&g_ctfs[pixel]);
				ref_imag *= __ldg(&g_ctfs[pixel]);
			}
			else
			{
				ref_real *= part_scale;
				ref_imag *= part_scale;
			}

			img_real = __ldg(&g_img_real[pixel]);
			img_imag = __ldg(&g_img_imag[pixel]);

			for (unsigned long itrans = 0; itrans < translation_num; itrans++)
			{
				XFLOAT weight = __ldg(&g_weights[bid * translation_num + itrans]);

				if (weight >= significant_weight)
				{
					weight /= weight_norm;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
					else
						translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

					XFLOAT diff_real = ref_real - trans_real;
					XFLOAT diff_imag = ref_imag - trans_imag;

					s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);

					s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
					s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
				}
			}

			cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
			cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
			cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
		}
	}
}

template<bool REFCTF, bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_wavg(
		XFLOAT *g_eulers,
		AccProjectorKernel projector,
		unsigned image_size,
		unsigned long orientation_num,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
        XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		XFLOAT *g_wdiff2s_parts,
		XFLOAT *g_wdiff2s_AA,
		XFLOAT *g_wdiff2s_XA,
        XFLOAT *g_model_var,
        XFLOAT *g_model_weight,
		unsigned long translation_num,
		XFLOAT weight_norm,
		XFLOAT significant_weight,
		XFLOAT part_scale,
        XFLOAT padding_factor,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz,
        int max_r2)
{
	XFLOAT ref_real, ref_imag, img_real, img_imag, trans_real, trans_imag;

	int bid = blockIdx.x; //block ID
	int tid = threadIdx.x;

	extern __shared__ XFLOAT buffer[];

	unsigned pass_num(ceilfracf(image_size,block_sz)),pixel;
	XFLOAT * s_wdiff2s_parts	= &buffer[0];
	XFLOAT * s_sumXA			= &buffer[block_sz];
	XFLOAT * s_sumA2			= &buffer[2*block_sz];
	XFLOAT * s_eulers           = &buffer[3*block_sz];

	if (tid < 9)
		s_eulers[tid] = g_eulers[bid*9+tid];
	__syncthreads();

	for (unsigned pass = 0; pass < pass_num; pass++) // finish a reference proj in each block
	{
		s_wdiff2s_parts[tid] = 0.0f;
		s_sumXA[tid] = 0.0f;
		s_sumA2[tid] = 0.0f;

		pixel = pass * block_sz + tid;

		if(pixel<image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}
            if(x*x + y*y > projector.maxR*projector.maxR) continue;

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					s_eulers[0], s_eulers[1], s_eulers[2],
					s_eulers[3], s_eulers[4], s_eulers[5],
					s_eulers[6], s_eulers[7], s_eulers[8],
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					s_eulers[6], s_eulers[7],
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
						x,y,
					s_eulers[0], s_eulers[1],
					s_eulers[3], s_eulers[4],
					ref_real, ref_imag);

            XFLOAT ctf, minvsigma2;
            minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);

			if (REFCTF)
			{
                ctf = __ldg(&g_ctfs[pixel]);
				ref_real *= ctf;//__ldg(&g_ctfs[pixel]);
				ref_imag *= ctf;//__ldg(&g_ctfs[pixel]);
			}
			else
			{
				ref_real *= part_scale;
				ref_imag *= part_scale;
			}

			img_real = __ldg(&g_img_real[pixel]);
			img_imag = __ldg(&g_img_imag[pixel]);
            
            XFLOAT tot_weight = 0.;
			for (unsigned long itrans = 0; itrans < translation_num; itrans++)
			{
				XFLOAT weight = __ldg(&g_weights[bid * translation_num + itrans]);

				if (weight >= significant_weight)
				{
					weight /= weight_norm;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, trans_real, trans_imag);
					else
						translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, trans_real, trans_imag);

					XFLOAT diff_real = ref_real - trans_real;
					XFLOAT diff_imag = ref_imag - trans_imag;
                    tot_weight += weight * weight * (diff_real*diff_real + diff_imag*diff_imag);

					s_wdiff2s_parts[tid] += weight * (diff_real*diff_real + diff_imag*diff_imag);

					s_sumXA[tid] +=  weight * ( ref_real * trans_real + ref_imag * trans_imag);
					s_sumA2[tid] +=  weight * ( ref_real*ref_real  +  ref_imag*ref_imag );
				}
			}

			cuda_atomic_add(&g_wdiff2s_XA[pixel], s_sumXA[tid]);
			cuda_atomic_add(&g_wdiff2s_AA[pixel], s_sumA2[tid]);
			cuda_atomic_add(&g_wdiff2s_parts[pixel], s_wdiff2s_parts[tid]);
            //backproject
            if(tot_weight > 0.){
                if ( ( x * x + y * y ) > max_r2)
				continue;
                tot_weight *= ctf * ctf;
                tot_weight *= minvsigma2 * minvsigma2;

                XFLOAT xp,yp,zp;
                if(DATA3D)
                {
                    xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
                    yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
                    zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
                }
                else
                {
                    xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
                    yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
                    zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
                }
                // Only asymmetric half is stored
                if (xp < (XFLOAT) 0.0)
                {
                    // Get complex conjugated hermitian symmetry pair
                    xp = -xp;
                    yp = -yp;
                    zp = -zp;
                }

                int x0 = floorf(xp);
                XFLOAT fx = xp - x0;
                int x1 = x0 + 1;

                int y0 = floorf(yp);
                XFLOAT fy = yp - y0;
                y0 -= mdl_inity;
                int y1 = y0 + 1;

                int z0 = floorf(zp);
                XFLOAT fz = zp - z0;
                z0 -= mdl_initz;
                int z1 = z0 + 1;

                XFLOAT mfx = (XFLOAT)1.0 - fx;
                XFLOAT mfy = (XFLOAT)1.0 - fy;
                XFLOAT mfz = (XFLOAT)1.0 - fz;

                XFLOAT dd000 = mfz * mfy * mfx;

                cuda_atomic_add(&g_model_var  [z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * tot_weight);

                XFLOAT dd001 = mfz * mfy *  fx;

                cuda_atomic_add(&g_model_var  [z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * tot_weight);

                XFLOAT dd010 = mfz *  fy * mfx;

                cuda_atomic_add(&g_model_var  [z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * tot_weight);

                XFLOAT dd011 = mfz *  fy *  fx;

                cuda_atomic_add(&g_model_var  [z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * tot_weight);

                XFLOAT dd100 =  fz * mfy * mfx;

                cuda_atomic_add(&g_model_var  [z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * tot_weight);

                XFLOAT dd101 =  fz * mfy *  fx;

                cuda_atomic_add(&g_model_var  [z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * tot_weight);

                XFLOAT dd110 =  fz *  fy * mfx;

                cuda_atomic_add(&g_model_var  [z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * tot_weight);

                XFLOAT dd111 =  fz *  fy *  fx;

                cuda_atomic_add(&g_model_var  [z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * tot_weight);
                //cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * tot_weight);
            }

        }
	}
}
#endif /* CUDA_WAVG_KERNEL_CUH_ */
