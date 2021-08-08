#pragma once
#include <torch/torch.h>
#include <utility>
#include "src/image.h"
#include "src/multidim_array.h"

struct RealReconstructorImpl : public torch::nn::Module 
{
    public:
        torch::Tensor volume; 
        torch::Tensor rot_grid;
        int vol_size;
        bool do_damping;
        torch::Tensor ctf_angle;
        torch::Tensor u2;
        torch::Tensor u4;
        float angpix;
        int device;

        RealReconstructorImpl(int vol_size_, bool do_damping_, float angpix_, int device_)
        {
            vol_size = vol_size_;
            device   = device_;
            std::vector<long int> vol_dim = {vol_size, vol_size, vol_size};
            angpix = angpix_;
            do_damping = do_damping_;
            torch::Device dev(torch::kCUDA, device);

            //volume = torch::from_blob(vol_.data(), c10::ArrayRef<long int>(vol_dim), 
            //       torch::dtype(torch::kFloat32).requires_grad(true));

            volume = torch::zeros(c10::ArrayRef<long int>(vol_dim), torch::dtype(torch::kFloat32).requires_grad(true));
            //std::cout << "volume requires grad: " << volume.requires_grad() << std::endl;
            
            //initialise static tensors for ctf computation
            torch::Tensor x_idx = torch::arange(vol_size/2+1)/float(vol_size*angpix);//torch::arange(0, 6);
            torch::Tensor y_idx = torch::arange(-vol_size/2 + 1, vol_size/2 + 1)/float(vol_size*angpix);
            auto grid = torch::meshgrid({y_idx, x_idx});
            //grid[0] corresponding to y, where grid[1] corresponding to x
            auto gridy = grid[0].to(dev);///ys;//ip [-V+1, V)
            auto gridx = grid[1].to(dev);///xs;//jp [0, V)
            //grid[1] /= xs;
            //shift to fft freqs
            gridy = torch::roll(gridy, {vol_size/2+1}, {0});
            ctf_angle = torch::atan2(gridy, gridx).to(dev);
            u2 = gridx*gridx + gridy*gridy;
            u4 = u2*u2;
        }

        void set_volume(MultidimArray<float> & vol_, bool logging = false)
        {
            torch::NoGradGuard no_grad;
            std::cout << "Iref size: " << MULTIDIM_SIZE(vol_) << " " << volume.numel() << " " << MULTIDIM_SIZE(vol_) - volume.numel() << std::endl;
            volume = volume.to(torch::kCPU);
            std::vector<long int> vol_dim = {vol_size, vol_size, vol_size};
            volume = torch::from_blob(vol_.data, c10::ArrayRef<long int>(vol_dim), torch::dtype(torch::kFloat32)).clone();

            //std::memcpy(reinterpret_cast<void *>(volume.data_ptr()), reinterpret_cast<void *> (vol_.data), volume.numel() * sizeof(torch::kFloat32));
            //std::copy(volume.data_ptr(), 
            //std::memcpy((void *)volume.data_ptr(), (void *)(vol_.data), MULTIDIM_SIZE(vol_) * sizeof(torch::kFloat32));
            torch::Device dev(torch::kCUDA, device);
            volume = volume.to(dev);

            if(logging)
            {
                save_volume("tmp_vol.mrc");
                //save_vector(vol_, "tmp_vec.mrc");
            }
        }

        void save_volume(std::string file_name)
        {
            Image<float> tmp_vol(vol_size, vol_size, vol_size);
            tmp_vol().setXmippOrigin();
            auto cpu_vol = volume.to(torch::kCPU);
            float * cpu_vol_ptr = cpu_vol.data_ptr<float>();
            //std::memcpy((void *)tmp_vol().data, (void *)cpu_vol.data_ptr(), sizeof(torch::kFloat32) * cpu_vol.numel());
            std::copy(cpu_vol_ptr, cpu_vol_ptr + MULTIDIM_SIZE(tmp_vol()), tmp_vol().data);
            tmp_vol.write(file_name);
        }

        void save_vector(MultidimArray<float>& vol, std::string file_name)
        {
            Image<float> tmp_vol(vol_size, vol_size, vol_size);
            //tmp_vol() = vol;
            //std::memcpy(reinterpret_cast<void *> (tmp_vol().data), reinterpret_cast<void *> (vol.data), MULTIDIM_SIZE(vol) * sizeof(torch::kFloat32));
            std::copy(vol.data, vol.data + MULTIDIM_SIZE(vol), tmp_vol().data);
            tmp_vol.write(file_name);
        }

        torch::Tensor forward(torch::Tensor thetas)
        {
            //generate sampling grid according to thetas
            auto rot_mat = angles_to_mat(thetas);

            //create affine grid using rotation matrices
            rot_grid = torch::nn::functional::affine_grid(rot_mat, {thetas.sizes()[0], 1, vol_size, vol_size, vol_size});

            //now transform the 3d volume using the affine grid
            using namespace torch::indexing;
            auto rotated_volume = torch::nn::functional::grid_sample(volume.view({1, 1, vol_size, vol_size, vol_size}), 
                    rot_grid.index({Slice(0, 1)}), 
                    torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(false));
            auto projections = torch::sum(rotated_volume, 2);
            for(int i = 1; i < rot_grid.size(0); i++)
            {
                auto tmp = torch::nn::functional::grid_sample(volume.view({1, 1, vol_size, vol_size, vol_size}), 
                    rot_grid.index({Slice(i, i+1)}), 
                    torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(false));

                auto proj = torch::sum(tmp, 2);

                projections = torch::cat({projections, proj}, 0);
            }

            //summation along z, the dim of input is n c d h w, the dimension of result is n c h w
            //auto projections = torch::sum(rotated_volume, 2);

            //center fft
            //auto projections_centered = torch::roll(projections, {vol_size/2, vol_size/2}, {2, 3});

            //get fourier transform of projections, and apply ctf
            //auto fft_projections = torch::fft::rfft2(projections);
            return projections;

        }

        torch::Tensor compute_ctf(std::vector<float>& deltas, float deltaa, std::vector<float>& Ks, float Q0)
        {
            float defocus_average = -(deltas[0] + deltas[1])*0.5;
            float defocus_deviation = -(deltas[0] - deltas[1])*0.5;
            auto deltaf = defocus_average + defocus_deviation*torch::cos(2.*(ctf_angle - deltaa));
            auto argument = Ks[0]*deltaf*u2 + Ks[1]*u4 - Ks[4];
            auto ctf = -(Ks[2]*torch::sin(argument) - Q0*torch::cos(argument));
            if(do_damping && Ks[3] != 0) 
            {
                auto scale = torch::exp(Ks[3] * u2);
                ctf *= scale;
            }
            return ctf;

        }

        torch::Tensor angles_to_mat(torch::Tensor thetas)
        {
            //thetas = torch::deg2rad(thetas);
            
            torch::Tensor rot_mat = torch::zeros({thetas.size(0), 3, 4}, torch::TensorOptions().dtype(torch::kFloat32));

            using namespace torch::indexing;

            auto ca = torch::cos(thetas.index({Slice(), 0}));
            auto cb = torch::cos(thetas.index({Slice(), 1}));
            auto cg = torch::cos(thetas.index({Slice(), 2}));

            auto sa = torch::sin(thetas.index({Slice(), 0}));
            auto sb = torch::sin(thetas.index({Slice(), 1}));
            auto sg = torch::sin(thetas.index({Slice(), 2}));

            auto cc = cb * ca;
            auto cs = cb * sa;
            auto sc = sb * ca;
            auto ss = sb * sa;

            //rot_mat.index_put_({Slice(), 0, 0}, cg * cc - sg * sa);
            //rot_mat.index_put_({Slice(), 0, 1}, cg * cs + sg * ca);
            //rot_mat.index_put_({Slice(), 0, 2}, -cg * sb);
            //rot_mat.index_put_({Slice(), 1, 0}, -sg * cc - cg * sa);
            //rot_mat.index_put_({Slice(), 1, 1}, -sg * cs + cg * ca);
            //rot_mat.index_put_({Slice(), 1, 2}, sg * sb);
            //rot_mat.index_put_({Slice(), 2, 0}, sc);
            //rot_mat.index_put_({Slice(), 2, 1}, ss);
            //rot_mat.index_put_({Slice(), 2, 2}, cb);

            rot_mat.index_put_({Slice(), 0, 0}, cg * cc - sg * sa);
            rot_mat.index_put_({Slice(), 0, 1}, -sg * cc - cg * sa);
            rot_mat.index_put_({Slice(), 0, 2}, sc);
            rot_mat.index_put_({Slice(), 1, 0}, cg * cs + sg * ca);
            rot_mat.index_put_({Slice(), 1, 1}, -sg * cs + cg * ca);
            rot_mat.index_put_({Slice(), 1, 2}, ss);
            rot_mat.index_put_({Slice(), 2, 0}, -cg * sb);
            rot_mat.index_put_({Slice(), 2, 1}, sg * sb);
            rot_mat.index_put_({Slice(), 2, 2}, cb);

            rot_mat = rot_mat.to(torch::Device(torch::kCUDA, device));

            return rot_mat;
        }

        void get_volume(std::vector<float> vol_out)
        {
            vol_out.resize(volume.numel());
            volume.to(torch::kCPU);
            std::copy(volume.data_ptr<float>(), volume.data_ptr<float>() + volume.numel(), vol_out.begin());
        }

};

TORCH_MODULE_IMPL(RealReconstructor, RealReconstructorImpl);
