#ifndef CUDA_LASSO_H_
#define CUDA_LASSO_H_
#include "src/acc/cuda/cuda_ml_optimiser.h"

void cuda_lasso(int fsc143, int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<Complex> &Ftest_conv, MultidimArray<RFLOAT> &Ftest_weight, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normfft, RFLOAT nrparts, bool do_nag = true, RFLOAT implicit_weight = 0.1, RFLOAT epsp = 0.01, RFLOAT bfactor = 2.);

void cuda_lasso_nocv(int fsc143, int tv_iters, RFLOAT l_r, RFLOAT mu, RFLOAT tv_alpha, RFLOAT tv_beta, RFLOAT eps, MultidimArray<RFLOAT> &Fconv,
        MultidimArray<RFLOAT> &Fweight, MultidimArray<Complex> &Fdata, MultidimArray<RFLOAT> &vol_out, MlDeviceBundle *devBundle, int data_dim, RFLOAT normalise, RFLOAT nrparts, bool do_nag = true, RFLOAT implicit_weight = 0.1, RFLOAT epsp = 0.01, RFLOAT bfactor=2.);
#endif
