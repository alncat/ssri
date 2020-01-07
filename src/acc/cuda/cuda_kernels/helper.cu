#include "src/acc/settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"
#include "src/acc/cuda/cuda_kernels/helper.cuh"
#include "src/acc/cuda/cuda_settings.h"

#include <curand.h>
#include <curand_kernel.h>

/// Needed explicit template instantiations
template __global__ void cuda_kernel_make_eulers_2D<true>(XFLOAT *,
	XFLOAT *, unsigned);
template __global__ void cuda_kernel_make_eulers_2D<false>(XFLOAT *,
	XFLOAT *, unsigned);

template __global__ void cuda_kernel_make_eulers_3D<true, true, true>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<true, true, false>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<true, false,true>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<true, false,false>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<false,true, true>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<false,true, false>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<false,false,true>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);
template __global__ void cuda_kernel_make_eulers_3D<false,false,false>(XFLOAT *,
		XFLOAT *, XFLOAT *, XFLOAT *, unsigned, XFLOAT *, XFLOAT *);

/*
 * This draft of a kernel assumes input that has jobs which have a single orientation and sequential translations within each job.
 *
 */
__global__ void cuda_kernel_exponentiate_weights_fine(
		XFLOAT *g_pdf_orientation,
		bool *g_pdf_orientation_zeros,
		XFLOAT *g_pdf_offset,
		bool *g_pdf_offset_zeros,
		XFLOAT *g_weights,
		XFLOAT min_diff2,
		int oversamples_orient,
		int oversamples_trans,
		unsigned long *d_rot_id,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num,
		long int job_num)
{
	// blockid
	int bid  = blockIdx.x;
	//threadid
	int tid = threadIdx.x;

	long int jobid = bid*SUMW_BLOCK_SIZE+tid;

	if (jobid<job_num)
	{
		long int pos = d_job_idx[jobid];
		// index of comparison
		long int ix = d_rot_id   [pos];   // each thread gets its own orient...
		long int iy = d_trans_idx[pos];   // ...and it's starting trans...
		long int in = d_job_num  [jobid]; // ...AND the number of translations to go through

		int c_itrans;
		for (int itrans=0; itrans < in; itrans++, iy++)
		{
			c_itrans = ( iy - (iy % oversamples_trans))/ oversamples_trans;

			if( g_weights[pos+itrans] < min_diff2 || g_pdf_orientation_zeros[ix] || g_pdf_offset_zeros[c_itrans])
				g_weights[pos+itrans] = -99e99; //large negative number
			else
				g_weights[pos+itrans] = g_pdf_orientation[ix] + g_pdf_offset[c_itrans] + min_diff2 - g_weights[pos+itrans];
		}
	}
}

__global__ void cuda_kernel_initRND(unsigned long seed, curandState *States)
{
       int tid = threadIdx.x;
       int bid = blockIdx.x;

       int id    = bid*RND_BLOCK_SIZE + tid;
       int pixel = bid*RND_BLOCK_SIZE + tid;

       curand_init(seed, pixel, 0, &States[id]);
}

__global__ void cuda_kernel_RNDnormalDitributionComplexWithPowerModulation2D( ACCCOMPLEX *Image,
																		    curandState *States,
																		    long int xdim,
																			XFLOAT * spectra)
{
       int tid = threadIdx.x;
       int bid = blockIdx.x;

       int id    = bid*RND_BLOCK_SIZE + tid;
       int pixel = bid*RND_BLOCK_SIZE + tid;

       //curand_init(1234, pixel, 0, &States[id]);

       int x,y;
       int size = xdim*((xdim-1)*2);   					//assuming square input images (particles)
       int passes = size/(RND_BLOCK_NUM*RND_BLOCK_SIZE) + 1;
       for(int i=0; i<passes; i++)
       {
               if(pixel<size)
               {
                       y = ( pixel / xdim );
                       x = pixel % xdim;

                       // fftshift in one of two dims;
                       if(y>=xdim)
                               y -= (xdim-1)*2;   		//assuming square input images (particles)

                       int ires = rintf(sqrtf(x*x + y*y));
#if defined(ACC_DOUBLE_PRECISION)
                       XFLOAT scale = 0.;
                       if(ires<xdim)
                               scale =  spectra[ires];

                       Image[pixel] = (curand_normal2_double(&States[id]))*scale;
#else
                       XFLOAT scale = 0.f;
                       if(ires<xdim)
                               scale =  spectra[ires];

                       Image[pixel] = (curand_normal2(&States[id]))*scale;
#endif
               }
               pixel += RND_BLOCK_NUM*RND_BLOCK_SIZE;
       }
}
__global__ void cuda_kernel_RNDnormalDitributionComplexWithPowerModulation3D( ACCCOMPLEX *Image,
																		    curandState *States,
																		    long int xdim,
                                                                            long int ydim,
																			XFLOAT * spectra)
{
       int tid = threadIdx.x;
       int bid = blockIdx.x;

       int id    = bid*RND_BLOCK_SIZE + tid;
       int pixel = bid*RND_BLOCK_SIZE + tid;

       //curand_init(1234, pixel, 0, &States[id]);

       int x,y,z,xydim(xdim*ydim);
       int size = xdim*((xdim-1)*2)*((xdim-1)*2);   		//assuming square input images (particles)
       int passes = size/(RND_BLOCK_NUM*RND_BLOCK_SIZE) + 1;
       for(int i=0; i<passes; i++)
       {
               if(pixel<size)
               {
            	   	   z = pixel / xydim;
                       y = ( pixel - (z*xydim) / xdim );
                       x = pixel % xdim;
                       // fftshift in two of three dims;
                       if(z>=xdim)
                    	   z -= (xdim-1)*2;					//assuming square input images (particles)
                       if(y>=xdim)
                           y -= (xdim-1)*2;					//assuming square input images (particles)


                       int ires = rintf(sqrtf(x*x + y*y + z*z));
#if defined(ACC_DOUBLE_PRECISION)
                       XFLOAT scale = 0.;
                       if(ires<xdim)
                               scale =  spectra[ires];

                       Image[pixel] = (curand_normal2_double(&States[id]))*scale;
#else
                       XFLOAT scale = 0.f;
                       if(ires<xdim)
                               scale =  spectra[ires];

                       Image[pixel] = (curand_normal2(&States[id]))*scale;
#endif
               }
               pixel += RND_BLOCK_NUM*RND_BLOCK_SIZE;
       }
}


//__global__ void cuda_kernel_exponentiate_weights_fine2(
//		XFLOAT *g_pdf_orientation,
//		XFLOAT *g_pdf_offset,
//		XFLOAT *g_weights,
//		XFLOAT avg_diff2,
//		int oversamples_orient,
//		int oversamples_trans,
//		unsigned long *d_rot_id,
//		unsigned long *d_trans_idx,
//		unsigned long *d_job_idx,
//		unsigned long *d_job_num,
//		long int job_num)
//{
//	// blockid
//	int bid  = blockIdx.x;
//	//threadid
//	int tid = threadIdx.x;
//
//	long int jobid = bid*SUMW_BLOCK_SIZE+tid;
//
//	if (jobid<job_num)
//	{
//		long int pos = d_job_idx[jobid];
//		// index of comparison
//		long int iy = d_trans_idx[ pos];
//		long int in =  d_job_num[jobid];
//
//		int c_itrans;
//		for (int itrans=0; itrans < in; itrans++, iy++)
//		{
//			XFLOAT a = g_weights[pos+itrans] + avg_diff2;
//
//#if defined(ACC_DOUBLE_PRECISION)
//			if (a < -700.)
//				g_weights[pos+itrans] = 0.;
//			else
//				g_weights[pos+itrans] = exp(a);
//#else
//			if (a < -88.)
//				g_weights[pos+itrans] = 0.f;
//			else
//				g_weights[pos+itrans] = expf(a);
//#endif
//		}
//	}
//}

__global__ void cuda_kernel_softMaskOutsideMap(	XFLOAT *vol,
												long int vol_size,
												long int xdim,
												long int ydim,
												long int zdim,
												long int xinit,
												long int yinit,
												long int zinit,
												bool do_Mnoise,
												XFLOAT radius,
												XFLOAT radius_p,
												XFLOAT cosine_width	)
{

		int tid = threadIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
		XFLOAT r, raisedcos;

		__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT    partial_sum[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT partial_sum_bg[SOFTMASK_BLOCK_SIZE];

		XFLOAT sum_bg_total =  (XFLOAT)0.0;

		long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE);
		int texel = tid;

		partial_sum[tid]=(XFLOAT)0.0;
		partial_sum_bg[tid]=(XFLOAT)0.0;
		if (do_Mnoise)
		{
			for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
			{
				XFLOAT x,y,z;
				if(texel<vol_size)
				{
					img_pixels[tid]=__ldg(&vol[texel]);

					z = floor( (float) texel                   / (float)((xdim)*(ydim)));
					y = floor( (XFLOAT)(texel-z*(xdim)*(ydim)) / (XFLOAT) xdim );
					x = texel - z*(xdim)*(ydim) - y*xdim;

					z-=zinit;
					y-=yinit;
					x-=xinit;

					r = sqrt(x*x + y*y + z*z);

					if (r < radius)
						continue;
					else if (r > radius_p)
					{
						partial_sum[tid]    += (XFLOAT)1.0;
						partial_sum_bg[tid] += img_pixels[tid];
					}
					else
					{
#if defined(ACC_DOUBLE_PRECISION)
						raisedcos = 0.5 + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
						raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
						partial_sum[tid] += raisedcos;
						partial_sum_bg[tid] += raisedcos * img_pixels[tid];
					}
				}
			}
		}

		__syncthreads();
		for(int j=(SOFTMASK_BLOCK_SIZE/2); j>0; j/=2)
		{
			if(tid<j)
			{
				partial_sum[tid] += partial_sum[tid+j];
				partial_sum_bg[tid] += partial_sum_bg[tid+j];
			}
			__syncthreads();
		}

		sum_bg_total  = partial_sum_bg[0] / partial_sum[0];


		texel = tid;
		for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
		{
			XFLOAT x,y,z;
			if(texel<vol_size)
			{
				img_pixels[tid]=__ldg(&vol[texel]);

				z =  floor( (float) texel                  / (float)((xdim)*(ydim)));
				y = floor( (XFLOAT)(texel-z*(xdim)*(ydim)) / (XFLOAT)  xdim         );
				x = texel - z*(xdim)*(ydim) - y*xdim;

				z-=zinit;
				y-=yinit;
				x-=xinit;

				r = sqrt(x*x + y*y + z*z);

				if (r < radius)
					continue;
				else if (r > radius_p)
					img_pixels[tid]=sum_bg_total;
				else
				{
#if defined(ACC_DOUBLE_PRECISION)
					raisedcos = 0.5  + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
					raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
					img_pixels[tid]= img_pixels[tid]*(1-raisedcos) + sum_bg_total*raisedcos;

				}
				vol[texel]=img_pixels[tid];
			}

		}
}

__global__ void cuda_kernel_softMaskBackgroundValue(	XFLOAT *vol,
														long int vol_size,
														long int xdim,
														long int ydim,
														long int zdim,
														long int xinit,
														long int yinit,
														long int zinit,
														XFLOAT radius,
														XFLOAT radius_p,
														XFLOAT cosine_width,
														XFLOAT *g_sum,
														XFLOAT *g_sum_bg)
{

		int tid = threadIdx.x;
		int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
		XFLOAT r, raisedcos;
		int x,y,z;
		__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT    partial_sum[SOFTMASK_BLOCK_SIZE];
		__shared__ XFLOAT partial_sum_bg[SOFTMASK_BLOCK_SIZE];

		long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
		int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

		partial_sum[tid]=(XFLOAT)0.0;
		partial_sum_bg[tid]=(XFLOAT)0.0;

		for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
		{
			if(texel<vol_size)
			{
				img_pixels[tid]=__ldg(&vol[texel]);

				z =   texel / (xdim*ydim) ;
				y = ( texel % (xdim*ydim) ) / xdim ;
				x = ( texel % (xdim*ydim) ) % xdim ;

				z-=zinit;
				y-=yinit;
				x-=xinit;

				r = sqrt(XFLOAT(x*x + y*y + z*z));

				if (r < radius)
					continue;
				else if (r > radius_p)
				{
					partial_sum[tid]    += (XFLOAT)1.0;
					partial_sum_bg[tid] += img_pixels[tid];
				}
				else
				{
#if defined(ACC_DOUBLE_PRECISION)
					raisedcos = 0.5 + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
					raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
					partial_sum[tid] += raisedcos;
					partial_sum_bg[tid] += raisedcos * img_pixels[tid];
				}
			}
		}

		cuda_atomic_add(&g_sum[tid]   , partial_sum[tid]);
		cuda_atomic_add(&g_sum_bg[tid], partial_sum_bg[tid]);
}


__global__ void cuda_kernel_cosineFilter(	XFLOAT *vol,
											long int vol_size,
											long int xdim,
											long int ydim,
											long int zdim,
											long int xinit,
											long int yinit,
											long int zinit,
											bool do_noise,
											XFLOAT *noise,
											XFLOAT radius,
											XFLOAT radius_p,
											XFLOAT cosine_width,
											XFLOAT bg_value)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;

//		vol.setXmippOrigin(); // sets xinit=xdim , also for y z
	XFLOAT r, raisedcos, defVal;
	int x,y,z;
	__shared__ XFLOAT     img_pixels[SOFTMASK_BLOCK_SIZE];

	long int texel_pass_num = ceilfracf(vol_size,SOFTMASK_BLOCK_SIZE*gridDim.x);
	int texel = bid*SOFTMASK_BLOCK_SIZE*texel_pass_num + tid;

	defVal = bg_value;
	for (int pass = 0; pass < texel_pass_num; pass++, texel+=SOFTMASK_BLOCK_SIZE) // loop the available warps enough to complete all translations for this orientation
	{
		if(texel<vol_size)
		{
			img_pixels[tid]=__ldg(&vol[texel]);

			z =   texel / (xdim*ydim) ;
			y = ( texel % (xdim*ydim) ) / xdim ;
			x = ( texel % (xdim*ydim) ) % xdim ;

			z-=zinit;
			y-=yinit;
			x-=xinit;

			r = sqrt(XFLOAT(x*x + y*y + z*z));

			if(do_noise)
				defVal = noise[texel];

			if (r < radius)
				continue;
			else if (r > radius_p)
				img_pixels[tid]=defVal;
			else
			{
#if defined(ACC_DOUBLE_PRECISION)
				raisedcos = 0.5  + 0.5  * cospi( (radius_p - r) / cosine_width );
#else
				raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width );
#endif
				img_pixels[tid]= img_pixels[tid]*(1-raisedcos) + defVal*raisedcos;

			}
			vol[texel]=img_pixels[tid];
		}

	}
}

__global__ void cuda_kernel_centerFFT_2D(XFLOAT *img_in,
										 int image_size,
										 int xdim,
										 int ydim,
										 int xshift,
										 int yshift)
{

	__shared__ XFLOAT buffer[CFTT_BLOCK_SIZE];
	int tid = threadIdx.x;
	int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
	long int image_offset = image_size*blockIdx.y;
//	int pixel_pass_num = ceilfracf(image_size, CFTT_BLOCK_SIZE);

//	for (int pass = 0; pass < pixel_pass_num; pass++, pixel+=CFTT_BLOCK_SIZE)
//	{
		if(pixel<(image_size/2))
		{
			int y = floorf((XFLOAT)pixel/(XFLOAT)xdim);
			int x = pixel % xdim;				// also = pixel - y*xdim, but this depends on y having been calculated, i.e. serial evaluation

			int yp = y + yshift;
			if (yp < 0)
				yp += ydim;
			else if (yp >= ydim)
				yp -= ydim;

			int xp = x + xshift;
			if (xp < 0)
				xp += xdim;
			else if (xp >= xdim)
				xp -= xdim;

			int n_pixel = yp*xdim + xp;

			buffer[tid]                    = img_in[image_offset + n_pixel];
			img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
			img_in[image_offset + pixel]   = buffer[tid];
		}
//	}
}

__global__ void cuda_kernel_centerFFT_3D(XFLOAT *img_in,
										 int image_size,
										 int xdim,
										 int ydim,
										 int zdim,
										 int xshift,
										 int yshift,
									 	 int zshift)
{

	__shared__ XFLOAT buffer[CFTT_BLOCK_SIZE];
	int tid = threadIdx.x;
	int pixel = threadIdx.x + blockIdx.x*CFTT_BLOCK_SIZE;
	long int image_offset = image_size*blockIdx.y;

		int xydim = xdim*ydim;
		if(pixel<(image_size/2))
		{
			int z = floorf((XFLOAT)pixel/(XFLOAT)(xydim));
			int xy = pixel % xydim;
			int y = floorf((XFLOAT)xy/(XFLOAT)xdim);
			int x = xy % xdim;

			int xp = x + xshift;
			if (xp < 0)
				xp += xdim;
			else if (xp >= xdim)
				xp -= xdim;

			int yp = y + yshift;
			if (yp < 0)
				yp += ydim;
			else if (yp >= ydim)
				yp -= ydim;

			int zp = z + zshift;
			if (zp < 0)
				zp += zdim;
			else if (zp >= zdim)
				zp -= zdim;

			int n_pixel = zp*xydim + yp*xdim + xp;

			buffer[tid]                    = img_in[image_offset + n_pixel];
			img_in[image_offset + n_pixel] = img_in[image_offset + pixel];
			img_in[image_offset + pixel]   = buffer[tid];
		}
}


__global__ void cuda_kernel_probRatio(  XFLOAT *d_Mccf,
										XFLOAT *d_Mpsi,
										XFLOAT *d_Maux,
										XFLOAT *d_Mmean,
										XFLOAT *d_Mstddev,
										int image_size,
										XFLOAT normfft,
										XFLOAT sum_ref_under_circ_mask,
										XFLOAT sum_ref2_under_circ_mask,
										XFLOAT expected_Pratio,
										int NpsiThisBatch,
										int startPsi,
										int totalPsis)
{
	/* PLAN TO:
	 *
	 * 1) Pre-filter
	 * 		d_Mstddev[i] = 1 / (2*d_Mstddev[i])   ( if d_Mstddev[pixel] > 1E-10 )
	 * 		d_Mstddev[i] = 1    				  ( else )
	 *
	 * 2) Set
	 * 		sum_ref2_under_circ_mask /= 2.
	 *
	 * 3) Total expression becomes
	 * 		diff2 = ( exp(k) - 1.f ) / (expected_Pratio - 1.f)
	 * 	  where
	 * 	  	k = (normfft * d_Maux[pixel] + d_Mmean[pixel] * sum_ref_under_circ_mask)*d_Mstddev[i] + sum_ref2_under_circ_mask
	 *
	 */

	int pixel = threadIdx.x + blockIdx.x*(int)PROBRATIO_BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT Kccf = d_Mccf[pixel];
		XFLOAT Kpsi =(XFLOAT)-1.0;
		for(int psi = 0; psi < NpsiThisBatch; psi++ )
		{
			XFLOAT diff2 = normfft * d_Maux[pixel + image_size*psi];
			diff2 += d_Mmean[pixel] * sum_ref_under_circ_mask;

	//		if (d_Mstddev[pixel] > (XFLOAT)1E-10)
			diff2 *= d_Mstddev[pixel];
			diff2 += sum_ref2_under_circ_mask;

#if defined(ACC_DOUBLE_PRECISION)
			diff2 = exp(-diff2 / 2.); // exponentiate to reflect the Gaussian error model. sigma=1 after normalization, 0.4=1/sqrt(2pi)
#else
			diff2 = expf(-diff2 / 2.f);
#endif

			// Store fraction of (1 - probability-ratio) wrt  (1 - expected Pratio)
			diff2 = (diff2 - (XFLOAT)1.0) / (expected_Pratio - (XFLOAT)1.0);
			if (diff2 > Kccf)
			{
				Kccf = diff2;
				Kpsi = (startPsi + psi)*(360/totalPsis);
			}
		}
		d_Mccf[pixel] = Kccf;
		if (Kpsi >= 0.)
			d_Mpsi[pixel] = Kpsi;
	}
}

__global__ void cuda_kernel_rotateOnly(   ACCCOMPLEX *d_Faux,
						  	  	  	  	  XFLOAT psi,
						  	  			  AccProjectorKernel projector,
						  	  			  int startPsi
						  	  			  )
{
	int proj = blockIdx.y;
	int image_size=projector.imgX*projector.imgY;
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		int y = floorfracf(pixel,projector.imgX);
		int x = pixel % projector.imgX;

		if (y > projector.maxR)
		{
			if (y >= projector.imgY - projector.maxR)
				y = y - projector.imgY;
			else
				x = projector.maxR;
		}

		XFLOAT sa, ca;
		sincos((proj+startPsi)*psi, &sa, &ca);
		ACCCOMPLEX val;

		projector.project2Dmodel(	 x,y,
									 ca,
									-sa,
									 sa,
									 ca,
									 val.x,val.y);

		long int out_pixel = proj*image_size + pixel;

		d_Faux[out_pixel].x =val.x;
		d_Faux[out_pixel].y =val.y;
	}
}

__global__ void cuda_kernel_rotateAndCtf( ACCCOMPLEX *d_Faux,
						  	  	  	  	  XFLOAT *d_ctf,
						  	  	  	  	  XFLOAT psi,
						  	  			  AccProjectorKernel projector,
						  	  			  int startPsi
						  	  			  )
{
	int proj = blockIdx.y;
	int image_size=projector.imgX*projector.imgY;
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		int y = floorfracf(pixel,projector.imgX);
		int x = pixel % projector.imgX;

		if (y > projector.maxR)
		{
			if (y >= projector.imgY - projector.maxR)
				y = y - projector.imgY;
			else
				x = projector.maxR;
		}

		XFLOAT sa, ca;
		sincos((proj+startPsi)*psi, &sa, &ca);
		ACCCOMPLEX val;

		projector.project2Dmodel(	 x,y,
									 ca,
									-sa,
									 sa,
									 ca,
									 val.x,val.y);

		long int out_pixel = proj*image_size + pixel;

		d_Faux[out_pixel].x =val.x*d_ctf[pixel];
		d_Faux[out_pixel].y =val.y*d_ctf[pixel];

	}
}


__global__ void cuda_kernel_convol_A( ACCCOMPLEX *d_A,
									 ACCCOMPLEX *d_B,
									 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel].x;
		XFLOAT ti = - d_A[pixel].y;
		d_A[pixel].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_A[pixel].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_A( ACCCOMPLEX *d_A,
									 ACCCOMPLEX *d_B,
									 ACCCOMPLEX *d_C,
									 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel].x;
		XFLOAT ti = - d_A[pixel].y;
		d_C[pixel].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_C[pixel].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_A( ACCCOMPLEX *d_A,
									 	 	ACCCOMPLEX *d_B,
									 	 	int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel + A_off].x;
		XFLOAT ti = - d_A[pixel + A_off].y;
		d_A[pixel + A_off].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_A[pixel + A_off].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_A( ACCCOMPLEX *d_A,
									 	 	ACCCOMPLEX *d_B,
									 	 	ACCCOMPLEX *d_C,
									 	 	int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr =   d_A[pixel + A_off].x;
		XFLOAT ti = - d_A[pixel + A_off].y;
		d_C[pixel + A_off].x =   tr*d_B[pixel].x - ti*d_B[pixel].y;
		d_C[pixel + A_off].y =   ti*d_B[pixel].x + tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_B(	 ACCCOMPLEX *d_A,
									 	 ACCCOMPLEX *d_B,
									 	 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel].x;
		XFLOAT ti = d_A[pixel].y;
		d_A[pixel].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_A[pixel].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_convol_B(	 ACCCOMPLEX *d_A,
									 	 ACCCOMPLEX *d_B,
									 	 ACCCOMPLEX *d_C,
									 	 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel].x;
		XFLOAT ti = d_A[pixel].y;
		d_C[pixel].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_C[pixel].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_convol_B(	 ACCCOMPLEX *d_A,
									 	 	 ACCCOMPLEX *d_B,
									 	 	 int image_size)
{
	long int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	int A_off = blockIdx.y*image_size;
	if(pixel<image_size)
	{
		XFLOAT tr = d_A[pixel + A_off].x;
		XFLOAT ti = d_A[pixel + A_off].y;
		d_A[pixel + A_off].x =   tr*d_B[pixel].x + ti*d_B[pixel].y;
		d_A[pixel + A_off].y =   ti*d_B[pixel].x - tr*d_B[pixel].y;
	}
}

__global__ void cuda_kernel_batch_multi( XFLOAT *A,
								   XFLOAT *B,
								   XFLOAT *OUT,
								   XFLOAT S,
		  	  	  	  	  	  	   int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		OUT[pixel + blockIdx.y*image_size] = A[pixel + blockIdx.y*image_size]*B[pixel + blockIdx.y*image_size]*S;
}

__global__ void cuda_kernel_finalizeMstddev( XFLOAT *Mstddev,
											 XFLOAT *aux,
											 XFLOAT S,
											 int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
	{
		XFLOAT temp = Mstddev[pixel] + S * aux[pixel];
		if(temp > 0)
			Mstddev[pixel] = sqrt(temp);
		else
			Mstddev[pixel] = 0;
	}
}

__global__ void cuda_kernel_square(
		XFLOAT *A,
		int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		A[pixel] = A[pixel]*A[pixel];
}

template<bool invert>
__global__ void cuda_kernel_make_eulers_2D(
		XFLOAT *alphas,
		XFLOAT *eulers,
		unsigned orientation_num)
{
	unsigned oid = blockIdx.x * BLOCK_SIZE + threadIdx.x; //Orientation id

	if (oid >= orientation_num)
		return;

	XFLOAT ca, sa;
	XFLOAT a = alphas[oid] * (XFLOAT)PI / (XFLOAT)180.0;

#ifdef ACC_DOUBLE_PRECISION
	sincos(a, &sa, &ca);
#else
	sincosf(a, &sa, &ca);
#endif

	if(!invert)
	{
		eulers[9 * oid + 0] = ca;//00
		eulers[9 * oid + 1] = sa;//01
		eulers[9 * oid + 2] = 0 ;//02
		eulers[9 * oid + 3] =-sa;//10
		eulers[9 * oid + 4] = ca;//11
		eulers[9 * oid + 5] = 0 ;//12
		eulers[9 * oid + 6] = 0 ;//20
		eulers[9 * oid + 7] = 0 ;//21
		eulers[9 * oid + 8] = 1 ;//22
	}
	else
	{
		eulers[9 * oid + 0] = ca;//00
		eulers[9 * oid + 1] =-sa;//10
		eulers[9 * oid + 2] = 0 ;//20
		eulers[9 * oid + 3] = sa;//01
		eulers[9 * oid + 4] = ca;//11
		eulers[9 * oid + 5] = 0 ;//21
		eulers[9 * oid + 6] = 0 ;//02
		eulers[9 * oid + 7] = 0 ;//12
		eulers[9 * oid + 8] = 1 ;//22
	}
}

template<bool invert, bool doL, bool doR>
__global__ void cuda_kernel_make_eulers_3D(
		XFLOAT *alphas,
		XFLOAT *betas,
		XFLOAT *gammas,
		XFLOAT *eulers,
		unsigned orientation_num,
		XFLOAT *L,
		XFLOAT *R)
{
	XFLOAT a(0.f),b(0.f),g(0.f), A[9],B[9];
	XFLOAT ca, sa, cb, sb, cg, sg, cc, cs, sc, ss;

	unsigned oid = blockIdx.x * BLOCK_SIZE + threadIdx.x; //Orientation id

	if (oid >= orientation_num)
		return;

	for (int i = 0; i < 9; i ++)
		B[i] = (XFLOAT) 0.f;

	a = alphas[oid] * (XFLOAT)PI / (XFLOAT)180.0;
	b = betas[oid]  * (XFLOAT)PI / (XFLOAT)180.0;
	g = gammas[oid] * (XFLOAT)PI / (XFLOAT)180.0;

#ifdef ACC_DOUBLE_PRECISION
	sincos(a, &sa, &ca);
	sincos(b,  &sb, &cb);
	sincos(g, &sg, &cg);
#else
	sincosf(a, &sa, &ca);
	sincosf(b,  &sb, &cb);
	sincosf(g, &sg, &cg);
#endif

	cc = cb * ca;
	cs = cb * sa;
	sc = sb * ca;
	ss = sb * sa;

	A[0] = ( cg * cc - sg * sa);//00
	A[1] = ( cg * cs + sg * ca);//01
	A[2] = (-cg * sb )         ;//02
	A[3] = (-sg * cc - cg * sa);//10
	A[4] = (-sg * cs + cg * ca);//11
	A[5] = ( sg * sb )         ;//12
	A[6] = ( sc )              ;//20
	A[7] = ( ss )              ;//21
	A[8] = ( cb )              ;//22

	if (doR)
	{
		for (int i = 0; i < 9; i++)
			B[i] = 0.f;

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++)
					B[i * 3 + j] += A[i * 3 + k] * R[k * 3 + j];
	}
	else
		for (int i = 0; i < 9; i++)
			B[i] = A[i];

	if (doL)
	{
		if (doR)
			for (int i = 0; i < 9; i++)
				A[i] = B[i];

		for (int i = 0; i < 9; i++)
			B[i] = 0.f;

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++)
					B[i * 3 + j] += L[i * 3 + k] * A[k * 3 + j];
	}

	if(invert)
	{
		eulers[9 * oid + 0] = B[0];//00
		eulers[9 * oid + 1] = B[3];//01
		eulers[9 * oid + 2] = B[6];//02
		eulers[9 * oid + 3] = B[1];//10
		eulers[9 * oid + 4] = B[4];//11
		eulers[9 * oid + 5] = B[7];//12
		eulers[9 * oid + 6] = B[2];//20
		eulers[9 * oid + 7] = B[5];//21
		eulers[9 * oid + 8] = B[8];//22
	}
	else
	{
		eulers[9 * oid + 0] = B[0];//00
		eulers[9 * oid + 1] = B[1];//10
		eulers[9 * oid + 2] = B[2];//20
		eulers[9 * oid + 3] = B[3];//01
		eulers[9 * oid + 4] = B[4];//11
		eulers[9 * oid + 5] = B[5];//21
		eulers[9 * oid + 6] = B[6];//02
		eulers[9 * oid + 7] = B[7];//12
		eulers[9 * oid + 8] = B[8];//22
	}
}

__global__ void cuda_kernel_allweights_to_mweights(
		unsigned long * d_iorient,
		XFLOAT * d_allweights,
		XFLOAT * d_mweights,
		unsigned long orientation_num,
		unsigned long translation_num,
        int block_size
		)
{
	size_t idx = blockIdx.x * block_size + threadIdx.x;
	if (idx < orientation_num*translation_num)
		d_mweights[d_iorient[idx/translation_num] * translation_num + idx%translation_num] =
				d_allweights[idx/translation_num * translation_num + idx%translation_num];
                // TODO - isn't this just d_allweights[idx + idx%translation_num]?   Really?
}

__global__ void cuda_kernel_complex_multi( XFLOAT *A,
                                   XFLOAT *B,
                                   XFLOAT S,
                                   int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        A[pixel*2] *= B[pixel]*S;
        A[pixel*2+1] *= B[pixel]*S;
    }
}

__global__ void cuda_kernel_complex_multi( XFLOAT *A,
                                   XFLOAT *B,
                                   XFLOAT S,
                                   XFLOAT w,
                                   int Z,
                                   int Y,
                                   int X,
                                   int ZZ,
                                   int YY,
                                   int XX,
                                   int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        if(kp >= X) kp -= (Z);
        if(ip >= X) ip -= (Y);
        XFLOAT freq = kp*kp + ip*ip + jp*jp;
        freq = 1.;//39.4784176*freq/(X*X) + 1.;
        if(kp < XX && kp > -XX && ip < XX && ip > -XX && jp < XX) {
            if(kp < 0) kp += ZZ;
            if(ip < 0) ip += YY;
            int n_pixel = kp*(YY*XX) + ip*XX + jp;
            A[pixel*2] *= (B[n_pixel]*S + w*freq);
            A[pixel*2+1] *= (B[n_pixel]*S + w*freq);
        } else {
            //A[pixel*2] = 0.;
            //A[pixel*2+1] = 0.;
            A[pixel*2] *=w*freq;
            A[pixel*2+1] *=w*freq;
        }
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT l,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        A[pixel] -= (B[pixel] - l*C[pixel]);
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT l,
                                     int Z,
                                     int Y,
                                     int X,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        int hZ = Z >> 1;
        int hY = Y >> 1;
        int hX = X >> 1;
        if(kp >= hZ) kp += Z;
        if(ip >= hY) ip += Y;
        if(jp >= hX) jp += X;
        hY = Y << 1;
        hX = X << 1;
        int c_pixel = kp*hY*hX + ip*hX + jp;
        A[c_pixel] -= (B[c_pixel] - l*C[c_pixel]);
    }
}

__global__ void cuda_kernel_substract(XFLOAT *A,
                                     XFLOAT *B,
                                     XFLOAT *C,
                                     XFLOAT *vol_out,
                                     XFLOAT l,
                                     XFLOAT* sum,
                                     int Z,
                                     int Y,
                                     int X,
                                     int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size) {
        int kp = pixel / (Y*X);
        int ip = (pixel - kp * (Y*X))/X;
        int jp = pixel - kp * (Y*X) - ip * X;
        int hZ = Z >> 1;
        int hY = Y >> 1;
        int hX = X >> 1;
        if(kp >= hZ) kp += Z;
        if(ip >= hY) ip += Y;
        if(jp >= hX) jp += X;
        hY = Y << 1;
        hX = X << 1;
        int c_pixel = kp*hY*hX + ip*hX + jp;
        XFLOAT tmp = B[c_pixel] - vol_out[c_pixel];
        tmp -= A[c_pixel];
        A[c_pixel] -= (B[c_pixel] - l*C[c_pixel]);
        cuda_atomic_add(&sum[0], tmp*tmp);
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *grads,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT th = l_r*alpha/(eps+fabsf(img[pixel]));
        XFLOAT tmp = img[pixel];
        img[pixel] -=  l_r*grads[pixel];
        grads[pixel] = tmp;
        if(img[pixel] < th && img[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(img[pixel] >= th){
                img[pixel] -= th;
            } else {
                img[pixel] += th;
            }
        }
        grads[pixel] -= img[pixel];
    }
}

__global__ void cuda_kernel_soft_threshold(XFLOAT *img,
                                           XFLOAT *grads,
                                           XFLOAT l_r,
                                           XFLOAT alpha,
                                           XFLOAT eps,
                                           int X,
                                           int Y,
                                           int Z,
                                           int XX,
                                           int YY,
                                           int ZZ,
                                           int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int k = pixel/(YY*XX);
        int i = (pixel - k*YY*XX)/XX;
        int j = pixel - k*YY*XX - i*XX;
        int hZ = ZZ >> 1;
        int hY = YY >> 1;
        int hX = XX >> 1;
        int kl = k;
        int il = i;
        int jl = j;
        if(kl >= hZ){
            kl -= ZZ;
            k = kl + Z;
        }
        if(il >= hY){
            il -= YY;
            i = il + Y;
        }
        if(jl >= hX){
            jl -= XX;
            j = jl + X;
        }

        pixel = k*Y*X + i*X + j;
        XFLOAT th = l_r*alpha/(eps+fabsf(img[pixel]));
        XFLOAT tmp = img[pixel];
        img[pixel] -=  l_r*grads[pixel];
        //grads[pixel] = tmp;
        if(img[pixel] < th && img[pixel] > -th){
            img[pixel] = 0.;
        } else {
            if(img[pixel] >= th){
                img[pixel] -= th;
            } else {
                img[pixel] += th;
            }
        }
        //grads[pixel] -= img[pixel];
    }
}

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Y,
                                       int X,
                                       XFLOAT beta,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT val = img[pixel];
        int i = pixel/X;
        int j = pixel - i*X;
        int hY = Y>>1;
        int hX = X>>1;
        XFLOAT tmp = 0.;
        int il = i;// + hY;
        int jl = j;// + hX;
        if (il >= hY) il -= Y;
        if (jl >= hX) jl -= X;
        //il -= hY;
        //jl -= hX;
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        if( il < hY - 1){
            int ipp = il + 1;
            if(il < -1) ipp += Y;
            int loc = ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int jpp = jl + 1;
            if(jl < -1) jpp += X;
            int loc = i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if(norm > eps*eps){
            tmp /= sqrt(norm);
            gtmp += tmp*beta;
        } else {
            gtmp += tmp*beta/eps;
        }
        //got the norm of il - 1, jl
        if( il > -hY ){
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            val = img[ipm*X + j];
            tmp = img[pixel] - val;
            norm = tmp*tmp;
            if( jl < hX - 1){
                int jpp = jl + 1;
                if(jl < -1) jpp += X;
                int loc = ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (val - img_loc)*(val - img_loc);
            }
            if(norm > eps*eps){
                tmp /= sqrt(norm);
                gtmp += tmp*beta;
            } else {
                gtmp += tmp*beta/eps;
            }
        }
        //got the norm of il, jl - 1
        //il ranges from 0, hX - 1, -hX, -1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            val = img[i*X + jpm];
            tmp = img[pixel] - val;
            norm = tmp*tmp;
            if( il < hY - 1){
                int ipp = il + 1;
                if(il < -1) ipp += X;
                int loc = ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (val - img_loc)*(val - img_loc);
            }
            if(norm > eps*eps){
                tmp /= sqrt(norm);
                gtmp += tmp*beta;
            } else {
                gtmp += tmp*beta/eps;
            }
        }
        grads[pixel] += gtmp;
        //if( ip > -hY)
        //{
        //    int ipp = ip - 1;
        //    if(ip < 1) ipp += Y;
        //    int loc = ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip < hY - 1)
        //{
        //    int ipp = ip + 1;
        //    if(ip < -1) ipp += Y;
        //    int loc = ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( jp > -hX)
        //{
        //    int jpp = jp - 1;
        //    if(jp < 1) jpp += X;
        //    int loc = i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //if( jp < hX - 1)
        //{
        //    int jpp = jp + 1;
        //    if(jp < -1) jpp += X;
        //    int loc = i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //grads[pixel] += tmp*beta;
    }

}

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Z,
                                       int Y,
                                       int X,
                                       int ZZ,
                                       int YY,
                                       int XX,
                                       XFLOAT beta,
                                       XFLOAT epslog,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        int k = pixel/(YY*XX);
        int i = (pixel - k*YY*XX)/XX;
        int j = pixel - k*YY*XX - i*XX;
        int hZ = ZZ>>1;
        int hY = YY>>1;
        int hX = XX>>1;
        XFLOAT tmp = 0.;
        int kl = k ;//+ hZ;
        int il = i ;//+ hY;
        int jl = j ;//+ hX;
        if (kl >= hZ) {
            kl -= ZZ;
            k  += ZZ;
        }
        if (il >= hY) {
            il -= YY;
            i  += YY;
        }
        if (jl >= hX) {
            jl -= XX;
            j  += XX;
        }
        XFLOAT val = img[k*Y*X+i*X+j];
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        int kpp = kl + 1;
        if(kl < -1) kpp += Z;
        int ipp = il + 1;
        if(il < -1) ipp += Y;
        int jpp = jl + 1;
        if(jl < -1) jpp += X;

        if( kl < hZ - 1){
            int loc = kpp*Y*X + i*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( il < hY - 1){
            int loc = k*Y*X + ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int loc = k*Y*X + i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        norm = sqrt(norm);
        if(norm > eps){
            tmp /= norm;
        } else {
            tmp /= eps;
        }
        gtmp += tmp/(norm + epslog)*beta;
        //got the norm of kl-1, il, jl
        //kl - 1 >= -hZ
        if( kl > -hZ ){
            int kpm = kl - 1;
            //kl - 1 < 0
            if(kl < 1) kpm += Z;
            XFLOAT nval = img[kpm*Y*X + i*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            //il + 1 < hY
            if( il < hY - 1){
                int loc = kpm*Y*X + ipp*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = kpm*Y*X + i*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //got the norm of kl, il - 1, jl
        if( il > -hY ){
            norm = 0.;
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            XFLOAT nval = img[k*Y*X + ipm*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + ipm*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = k*Y*X + ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp*beta/(norm + epslog);
        }
        //got the norm of kl, il, jl - 1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            XFLOAT nval = img[k*Y*X + i*X + jpm];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + i*X + jpm;
                norm += (nval - img[loc])*(nval - img[loc]);
            }
            if( il < hY - 1){
                int loc = k*Y*X + ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //grads[pixel] += gtmp;
        grads[k*Y*X+i*X+j] += gtmp;
    }
}

__global__ void cuda_kernel_graph_grad(XFLOAT *img,
                                       XFLOAT *grads,
                                       int Z,
                                       int Y,
                                       int X,
                                       XFLOAT beta,
                                       XFLOAT epslog,
                                       XFLOAT eps,
                                       int image_size)
{
    int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if(pixel < image_size){
        XFLOAT val = img[pixel];
        int k = pixel/(Y*X);
        int i = (pixel - k*Y*X)/X;
        int j = pixel - k*Y*X - i*X;
        int hZ = Z>>1;
        int hY = Y>>1;
        int hX = X>>1;
        XFLOAT tmp = 0.;
        int kl = k ;//+ hZ;
        int il = i ;//+ hY;
        int jl = j ;//+ hX;
        if (kl >= hZ) kl -= Z;
        if (il >= hY) il -= Y;
        if (jl >= hX) jl -= X;
        //kl -= hZ;
        //il -= hY;
        //jl -= hX;
        XFLOAT norm = 0.;
        XFLOAT gtmp = 0.;
        int kpp = kl + 1;
        if(kl < -1) kpp += Z;
        int ipp = il + 1;
        if(il < -1) ipp += Y;
        int jpp = jl + 1;
        if(jl < -1) jpp += X;

        if( kl < hZ - 1){
            int loc = kpp*Y*X + i*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( il < hY - 1){
            int loc = k*Y*X + ipp*X + j;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        if( jl < hX - 1){
            int loc = k*Y*X + i*X + jpp;
            XFLOAT img_loc = img[loc];
            tmp += val - img_loc;
            norm += (val - img_loc)*(val - img_loc);
        }
        norm = sqrt(norm);
        if(norm > eps){
            tmp /= norm;
        } else {
            tmp /= eps;
        }
        gtmp += tmp/(norm + epslog)*beta;
        //got the norm of kl-1, il, jl
        if( kl > -hZ ){
            int kpm = kl - 1;
            if(kl < 1) kpm += Z;
            XFLOAT nval = img[kpm*Y*X + i*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( il < hY - 1){
                int loc = kpm*Y*X + ipp*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = kpm*Y*X + i*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        //got the norm of kl, il - 1, jl
        if( il > -hY ){
            norm = 0.;
            int ipm = il - 1;
            if(il < 1) ipm += Y;
            XFLOAT nval = img[k*Y*X + ipm*X + j];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + ipm*X + j;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            if( jl < hX - 1){
                int loc = k*Y*X + ipm*X + jpp;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp*beta/(norm + epslog);
        }
        //got the norm of kl, il, jl - 1
        if( jl > -hX ){
            int jpm = jl - 1;
            if(jl < 1) jpm += X;
            XFLOAT nval = img[k*Y*X + i*X + jpm];
            tmp = val - nval;
            norm = tmp*tmp;
            if( kl < hZ - 1){
                int loc = kpp*Y*X + i*X + jpm;
                norm += (nval - img[loc])*(nval - img[loc]);
            }
            if( il < hY - 1){
                int loc = k*Y*X + ipp*X + jpm;
                XFLOAT img_loc = img[loc];
                norm += (nval - img_loc)*(nval - img_loc);
            }
            norm = sqrt(norm);
            if(norm > eps){
                tmp /= norm;
            } else {
                tmp /= eps;
            }
            gtmp += tmp/(norm + epslog)*beta;
        }
        grads[pixel] += gtmp;
        //if( kp > -hZ)
        //{
        //    int kpp = kp - 1;
        //    if(kp < 1) kpp += Z;
        //    int loc = kpp*Y*X + i*X + j;
        //    tmp += val - img[loc];
        //}
        //if( kp < hZ - 1)
        //{
        //    int kpp = kp + 1;
        //    if(kp < -1) kpp += Z;
        //    int loc = kpp*Y*X + i*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip > -hY)
        //{
        //    int ipp = ip - 1;
        //    if(ip < 1) ipp += Y;
        //    int loc = k*Y*X + ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( ip < hY - 1)
        //{
        //    int ipp = ip + 1;
        //    if(ip < -1) ipp += Y;
        //    int loc = k*Y*X + ipp*X + j;
        //    tmp += val - img[loc];
        //}
        //if( jp > -hX)
        //{
        //    int jpp = jp - 1;
        //    if(jp < 1) jpp += X;
        //    int loc = k*Y*X + i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //if( jp < hX - 1)
        //{
        //    int jpp = jp + 1;
        //    if(jp < -1) jpp += X;
        //    int loc = k*Y*X + i*X + jpp;
        //    tmp += val - img[loc];
        //}
        //grads[pixel] += tmp*beta;
    }

}


