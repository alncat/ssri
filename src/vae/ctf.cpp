#include "ctf.h"
#include "api.h"
#include <vector>

void compute_ctf(torch::Tensor& ctf, std::vector<torch::Tensor>& deltas, torch::Tensor& deltaa, std::vector<float>& Ks, float Q0, 
        torch::Tensor& angle_, torch::Tensor& u2, torch::Tensor& u4, float angpix, bool do_damping) {
    torch::Tensor defocus_average = -(deltas[0] + deltas[1])*0.5;
    torch::Tensor defocus_deviation = -(deltas[0] - deltas[1])*0.5;
    torch::Tensor deltaf = defocus_average + defocus_deviation*torch::cos(2.*(angle_ - deltaa));
    auto argument = Ks[0]*deltaf*u2 + Ks[1]*u4 - Ks[4];
    ctf = -(Ks[2]*torch::sin(argument) - Q0*torch::cos(argument));
    if(do_damping) {
        auto scale = torch::exp(Ks[3] * u2);
        ctf *= scale;
    }
}

void optimize_ctf(std::vector<float>& image_data, std::vector<float>& ref_data, std::vector<float>& Fctf, float& delta_u, float& delta_v, float& angle, float defocus_res, std::vector<float> Ks, float Q0, int image_size, float angpix, bool do_damping, int current_size, bool do_logging) {
    torch::Tensor deltau = torch::tensor(delta_u, torch::requires_grad());
    torch::Tensor deltav = torch::tensor(delta_v, torch::requires_grad());
    torch::Tensor deltaa = torch::tensor(angle);//, torch::requires_grad());
    std::vector<torch::Tensor> deltas = {deltau, deltav};
    
    std::vector<long int> dimensions = {image_size, image_size};
    auto options = torch::dtype(torch::kFloat32);
    std::vector<float> image_slice(image_size*image_size, 0);
    std::vector<float> ref_slice(image_size*image_size, 0);

    std::copy(image_data.begin(), image_data.begin() + image_size*image_size, image_slice.begin());
    std::copy(ref_data.begin(), ref_data.begin() + image_size*image_size, ref_slice.begin());
    torch::Tensor image = torch::from_blob(image_slice.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor ref_image = torch::from_blob(ref_slice.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor ref_ctf = torch::from_blob(Fctf.data(), c10::ArrayRef<long int>({current_size, current_size/2+1, 2}), options);
    ref_ctf = torch::view_as_complex(ref_ctf);
    
    torch::Tensor full_ref_ctf = torch::zeros({image_size, image_size/2 + 1, 2}, options);
    full_ref_ctf = torch::view_as_complex(full_ref_ctf);
    
    // set the slice in full_ref_ctf
    using namespace torch::indexing;
    auto ctfs1 = ref_ctf.index({Slice(None, current_size/2 + 1), Slice()});
    full_ref_ctf.index_put_({Slice(None, current_size/2 + 1), Slice(None, current_size/2 + 1)}, ctfs1);
    auto ctfs2 = ref_ctf.index({Slice(current_size/2 + 1, current_size), Slice()});
    full_ref_ctf.index_put_({Slice(image_size - current_size/2 + 1, image_size), Slice(None, current_size/2 + 1)}, ctfs2);

    // get the inverse fourier transform

    auto ctf_image = torch::fft::irfft2(full_ref_ctf);
    auto fft_ctf_image = torch::fft::rfft2(ctf_image);


    auto ref_image_centered = torch::roll(ref_image, {image_size/2, image_size/2}, {0, 1});
    auto ref_image_fft = torch::fft::rfft2(ref_image_centered);
    auto image_centered = torch::roll(image, {image_size/2, image_size/2}, {0, 1});
    auto image_fft = torch::fft::rfft2(image_centered);

    auto aa = torch::real(ref_image_fft.conj() * ref_image_fft);
    auto xa = torch::real((image_fft.conj()*ref_image_fft));

    auto xas1 = torch::slice(xa, 0, 0, current_size/2 + 1);
    xas1 = torch::slice(xas1, 1, 0, current_size/2 + 1);
    if(do_logging) std::cout << xas1.sizes() << std::endl;
    auto xas2 = torch::slice(xa, 0, image_size - current_size/2 + 1, image_size);
    xas2 = torch::slice(xas2, 1, 0, current_size/2 + 1);
    auto xa_win = torch::cat({xas1, xas2}, 0);

    auto aas1 = torch::slice(ref_image_fft, 0, 0, current_size/2 + 1);
    aas1 = torch::slice(aas1, 1, 0, current_size/2 + 1);
    auto aas2 = torch::slice(ref_image_fft, 0, image_size - current_size/2 + 1, image_size);
    aas2 = torch::slice(aas2, 1, 0, current_size/2 + 1);
    auto aa_win = torch::cat({aas1, aas2}, 0);

    float xs = image_size*angpix;
    float ys = image_size*angpix;

    torch::Tensor ctf;
    torch::Tensor ref_image_ctf;
    torch::set_num_threads(15);

    //std::cout << "image dim: " << image_size << std::endl;
    torch::Tensor x_idx = torch::arange(current_size/2+1)/float(image_size*angpix);//torch::arange(0, 6);
    torch::Tensor y_idx = torch::arange(-current_size/2 + 1, current_size/2 + 1)/float(image_size*angpix);
    auto grid = torch::meshgrid({y_idx, x_idx});
    //grid[0] corresponding to y, where grid[1] corresponding to x
    auto gridy = grid[0];///ys;//ip [-V+1, V)
    auto gridx = grid[1];///xs;//jp [0, V)
    //grid[1] /= xs;
    //shift to fft freqs
    gridy = torch::roll(gridy, {current_size/2+1}, {0});
    torch::Tensor angle_ = torch::atan2(gridy, gridx);
    torch::Tensor u2 = gridx*gridx + gridy*gridy;
    torch::Tensor u4 = u2*u2;
    auto image_var = torch::mean(image*image);

    auto correlation_fn = [&](){
        torch::Tensor defocus_average = -(deltas[0] + deltas[1])*0.5;
        torch::Tensor defocus_deviation = -(deltas[0] - deltas[1])*0.5;
        
        torch::Tensor deltaf = defocus_average + defocus_deviation*torch::cos(2.*(angle_ - deltaa));
        //std::cout << grid[0] << std::endl;
        //std::cout << grid[1] << std::endl;
        
        auto argument = Ks[0]*deltaf*u2 + Ks[1]*u4 - Ks[4];
        ctf = -(Ks[2]*torch::sin(argument) - Q0*torch::cos(argument));
        if(do_damping) {
            auto scale = torch::exp(Ks[3] * u2);
            ctf *= scale;
        }
        //auto image_fft = torch::fft::rfftn(image);
        //fftshift
        //auto ref_image_centered = torch::roll(ref_image, {image_size/2, image_size/2}, {0, 1});
        //auto ref_image_fft = torch::fft::rfftn(ref_image_centered);
        //auto ref_image_fft_ctf = ref_image_fft*ctf;
        //ref_image_ctf = torch::fft::irfftn(ref_image_fft_ctf);
        //fftshift
        //ref_image_ctf = torch::roll(ref_image_ctf, {image_size/2, image_size/2}, {0, 1});
        //image = image.reshape({1,image_size*image_size});
        //ref_image_ctf = ref_image_ctf_rolled.reshape({1,image_size*image_size});
        
        auto cc = torch::mean(xa_win*ctf);
        //auto ref_image_ctf_var = torch::mean(ref_image_ctf*ref_image_ctf);
        //auto image_mean = torch::mean(image);
        //auto ref_image_ctf_mean = torch::mean(ref_image_ctf);

        //image_var -= image_mean*image_mean;
        //ref_image_ctf_var -= ref_image_ctf_mean*ref_image_ctf_mean;

        //cc = cc/ (image_var + 1e-8);
        auto restraint = defocus_deviation*defocus_deviation/(defocus_res*defocus_res*float(image_size*image_size));

        return -cc + restraint;
        //return -torch::nn::functional::cosine_similarity(image, ref_image_ctf);
        //return -1.;
    };
    //correlation_fn();

    //if(torch::sqrt(torch::mean((ctf - ref_ctf)*(ctf - ref_ctf))).item<float>() > 1e-2) {
    //    std::cout << "check!" << std::endl;
    //}
    torch::optim::LBFGS lbfgs_optimizer(deltas, torch::optim::LBFGSOptions().line_search_fn("strong_wolfe"));
    auto cost = [&](){
        lbfgs_optimizer.zero_grad();
        auto correlation = correlation_fn();//torch::sum(image*ref_image_ctf);
        //std::cout << "correlation: " << correlation << std::endl;
        //correlation.backward({}, c10::optional<bool>(true));
        correlation.backward();
        if(do_logging) {
            std::cout << deltas[0].grad() << " ";
            std::cout << deltas[1].grad() << std::endl;
            //std::cout << deltaa << std::endl;
        }
        return correlation;
    };

    if(do_logging) {
        std::cout << "delta_u: " << delta_u << std::endl;
        std::cout << "delta_v: " << delta_v << std::endl;
    }
    for (int i = 0; i < 1; i++) {
        //lbfgs_optimizer.zero_grad();
        //auto corr = correlation_fn();
        //corr.backward();
        //lbfgs_optimizer.step(cost);
        if(do_logging) {
            std::cout << "correlation: " << correlation_fn() << std::endl;
            std::cout << "delta_u: " << delta_u << " " << deltau << " " << deltas[0].grad() << std::endl;
            std::cout << "delta_v: " << delta_v << " " << deltav << " " << deltas[1].grad() << std::endl;
            torch::save(ctf, "ctf1989.pt");
            torch::save(ref_ctf, "refc1989.pt");
            torch::save(image, "im1989.pt");
            torch::save(ref_image, "ref1989.pt");
            torch::save(fft_ctf_image, "refi1989.pt");
        }
    }
}
