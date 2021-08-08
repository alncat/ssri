#pragma once
#include <vector>
#include "src/multidim_array.h"

void initialise_model_optimizer(int, int, int, float, int);
void initialise_model_optimizer(int, int, int, int, float, int, bool);
int train_a_batch(std::vector<float> &data, int part_id);
int train_unet(std::vector<float> &proj, std::vector<float>& data, std::vector<float>& weight, int part_id, std::vector<float> &wrapped_output);

int train_denoising_unet(std::vector<float> &proj, std::vector<float> &data, std::vector<float>& weight, int part_id, std::vector<float> &wrapped_data);

void optimize_ctf(std::vector<float>& image_data, std::vector<float>& ref_data, std::vector<float>& Fctf, float& delta_u, float& delta_v, float& angle, float defocus_res, std::vector<float> Ks, float Q0, int image_size, float angpix, bool do_damping, int current_size, bool do_logging);

void neural_volume_reconstruction(std::vector<float>& volume, int vol_size, std::vector<float>& reconstructed_vol);

//real constructor realted api
void initialise_real_reconstructor(int rank, int device, int mask_size, int full_size, bool do_damping, float angpix, float learning_rate);
void set_real_reconstructor_volume(MultidimArray<float>& vol_, bool logging);

void train_real_reconstructor(std::vector<float>& images, std::vector<float>& corr, std::vector<float>& ref_projections, std::vector<float>& weight,
        MultidimArray<float>& ref_ctf,
        int current_size, std::vector<float>& thetas, std::vector<float>& deltas, float deltaa, std::vector<float>& Ks, float Q0, int part_id);

