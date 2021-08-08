// Copyright 2020-present pytorch-cpp Authors
#include "vae.h"
#include <utility>
#include "api.h"
#include "src/image.h"
#include <sys/stat.h>
#include "nvol.h"
//convolution size: (input_size + 2*padding - kernel_size)/stride + 1
//convolution transpose size: (input_size - 1)*stride + output_padding - 2*padding + (kernel_size - 1) + 1

VAEImpl::VAEImpl(int64_t h_dim, int64_t output1, int64_t output2, int64_t output3, int64_t output4, int64_t z_dim)
    : fc1(output4*9, z_dim),
      fc2(output4*9, z_dim),
      fc3(z_dim, output4*9),
      bn(output4),
      cnv1(cnvBlock(1, output1, 6, 3)),
      cnv2(cnvBlock(output1, output2, 5, 2)),
      cnv3(cnvBlock(output2, output3, 4, 2)),
      cnv4(cnvBlock(output3, output4, 4, 2)),
      cnv5(cnvBlock(output4, output4, 4, 2)),
      uncnv5(uncnvBlock(output4, output4, 4, 2)),
      uncnv4(uncnvBlock(output4, output3, 4, 2)),
      uncnv3(uncnvBlock(output3, output2, 4, 2)),
      uncnv2(uncnvBlock(output2, output1, 5, 2)),
      uncnv1(uncnvBlock(output1, 1, 6, 3)),
      //cnv1(torch::nn::Conv2dOptions(1, output1, 6).stride(3)),//240->79
      //cnv2(torch::nn::Conv2dOptions(output1, output2, 5).stride(2)),//38
      //cnv3(torch::nn::Conv2dOptions(output2, output3, 4).stride(2)),//18
      //cnv4(torch::nn::Conv2dOptions(output3, output4, 4).stride(2)),//8.padding(1)),
      //cnv5(torch::nn::Conv2dOptions(output4, output4, 4).stride(2)),//3
      //cnv6(torch::nn::Conv2dOptions(output4, output4, 3).stride(1)),//2
      flt(torch::nn::FlattenOptions().start_dim(1)),
      //unflt(torch::nn::UnflattenOptions(1, {output4, h_dim, h_dim}))
      //uncnv5(torch::nn::ConvTranspose2dOptions(output4, output4, 4).stride(2)),//8
      //uncnv4(torch::nn::ConvTranspose2dOptions(output4, output3, 4).stride(2)),//18
      //uncnv3(torch::nn::ConvTranspose2dOptions(output3, output2, 4).stride(2)),//38
      //uncnv2(torch::nn::ConvTranspose2dOptions(output2, output1, 5).stride(2)),//79
      //uncnv1(torch::nn::ConvTranspose2dOptions(output1, 1, 6).stride(3)),//.padding(2)),//240
      output4_(output4)
      {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("cnv1", cnv1);
    register_module("cnv2", cnv2);
    register_module("cnv3", cnv3);
    register_module("cnv4", cnv4);
    register_module("cnv5", cnv5);
    register_module("uncnv1", uncnv1);
    register_module("uncnv2", uncnv2);
    register_module("uncnv3", uncnv3);
    register_module("uncnv4", uncnv4);
    register_module("uncnv5", uncnv5);
    register_module("bn", bn);
    register_module("flt", flt);
    //register_module("unflt", unflt);
}

std::pair<torch::Tensor, torch::Tensor> VAEImpl::encode(torch::Tensor x) {
    x = cnv1->forward(x);
    x = cnv2->forward(x);
    x = cnv3->forward(x);
    x = cnv4->forward(x);
    x = cnv5->forward(x);
    x = flt(x);
    return {fc1->forward(x), fc2->forward(x)};
}

torch::Tensor VAEImpl::reparameterize(torch::Tensor mu, torch::Tensor log_var) {
    if (is_training()) {
        auto std = log_var.div(2).exp_();
        auto eps = torch::randn_like(std);
        return eps.mul(std).add_(mu);
    } else {
        // During inference, return mean of the learned distribution
        // See https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
        return mu;
    }
}

torch::Tensor VAEImpl::decode(torch::Tensor z) {
    auto h = fc3->forward(z);
    h = h.view({-1,output4_,3,3});
    h = uncnv5->forward(h);
    h = uncnv4->forward(h);
    h = uncnv3->forward(h);
    h = uncnv2->forward(h);
    h = uncnv1->forward(h);
    return h;
}

VAEOutput VAEImpl::forward(torch::Tensor x) {
    auto encode_output = encode(x);
    auto mu = encode_output.first;
    auto log_var = encode_output.second;
    auto z = reparameterize(mu, log_var);
    auto x_reconstructed = decode(z);
    return {x_reconstructed, mu, log_var};
}

static std::unique_ptr<VAE> vae_model;
static std::unique_ptr<CUNet2d> unet_model;
static std::unique_ptr<torch::optim::AdamW> vae_optimizer;
static int vae_index = 0;
static int vae_image_size;
static int vae_mask_size;
static int vae_rank = 0;

void initialise_model_optimizer(int image_size, int h_dim, int z_dim, float learning_rate, int rank){
    vae_image_size = image_size;
    vae_model = std::make_unique<VAE>(1, 32, 32, 64, 64, z_dim);
    torch::Device device(torch::kCUDA, 0);
    (*vae_model)->to(device);
    vae_optimizer = std::make_unique<torch::optim::AdamW>((*vae_model)->parameters(), torch::optim::AdamWOptions(learning_rate));
    vae_rank = rank;
}

void initialise_model_optimizer(int image_size, int mask_size, int h_dim, int out_channels, float learning_rate, int rank, bool downsampling){
    vae_image_size = image_size;
    vae_mask_size = mask_size;
    if(downsampling) mask_size /= 2;
    //use 1 for denoising, 2 for optical flow
    unet_model = std::make_unique<CUNet2d>(2, out_channels, mask_size);
    torch::Device device(torch::kCUDA, 0);
    (*unet_model)->to(device);
    vae_optimizer = std::make_unique<torch::optim::AdamW>((*unet_model)->parameters(), torch::optim::AdamWOptions(learning_rate));
    vae_rank = rank;
    //check if model exists
    bool file_exists = false;
    std::string model_file = "model"+std::to_string(vae_rank)+".pt";
    std::string opt_file = "optimizer"+std::to_string(vae_rank)+".pt";
    struct stat buffer;
    //file_exists = stat(model_file.c_str(), &buffer) == 0;
    if(file_exists){
        std::cout << "loading model " << model_file << std::endl;
        torch::load(*unet_model, model_file);
        torch::load(*vae_optimizer, opt_file);
    }
}

int train_a_batch(std::vector<float> &data, int part_id){
    torch::set_num_threads(20);
    (*vae_model)->train();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = data.size()/(vae_image_size*vae_image_size);
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    torch::Tensor torch_real_projections = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    //auto data_mean = torch::mean(torch_real_projections, {1,2,3}, true, true);
    //auto data_std = std::get<0>(data_std_mean);
    //auto data_mean = std::get<1>(data_std_mean);
    //std::cout << data_std << std::endl;
    //torch_real_projections = (torch_real_projections - data_mean)/(data_std + 1e-4);
    auto output = (*(vae_model))->forward(torch_real_projections);
    //compute loss or consider apply ctf
    auto reconstruction_loss = torch::nn::functional::mse_loss(output.reconstruction, torch_real_projections, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
    auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());
    auto loss = reconstruction_loss + kl_divergence;
    if(vae_index == 0) vae_optimizer->zero_grad();
    loss.backward();
    if(vae_index != 0 && vae_index %50 == 0) {
        vae_optimizer->step();
        vae_optimizer->zero_grad();
    }
    vae_index++;
    //save some reconstructions here
    if(vae_index % 1000 == 0){
        //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
        std::cout << vae_index << "/"
            << "Reconstruction loss: "
            << reconstruction_loss.item<double>() / torch_real_projections.size(0)
            << ", KL-divergence: " << kl_divergence.item<double>() / torch_real_projections.size(0)
            << std::endl;
    }
    if(vae_index % 10000 == 0){
        //save model
        std::string model_path = "model" + std::to_string(vae_rank) + ".pt";
        torch::save(*vae_model, model_path);
    }
    if(vae_index % 5000 == 0){
        //save images
        auto input_image = torch_real_projections.index({0});
        Image<float> debug_img(vae_image_size, vae_image_size);
        memcpy(debug_img.data.data, input_image.data_ptr<float>(), input_image.numel()*sizeof(float));
        debug_img.write("tmp"+std::to_string(vae_rank)+"/input"+std::to_string(part_id)+".mrc");
        auto output_image =output.reconstruction.index({0});
        memcpy(debug_img.data.data, output_image.data_ptr<float>(), output_image.numel()*sizeof(float));
        debug_img.write("tmp"+std::to_string(vae_rank)+"/output"+std::to_string(part_id)+".mrc");
    }

}

int wrap_image(std::vector<float> &proj, std::vector<float> &data, std::vector<float>& wrapped_data){
    torch::Device device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = proj.size()/(vae_image_size*vae_image_size);
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    std::vector<long int> w_dims = {batch_num};
    torch::Tensor torch_real_projections = torch::from_blob(proj.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_data = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    int start_dim = vae_image_size/2 - vae_mask_size/2;
    int end_pad = vae_image_size - (start_dim + vae_mask_size);
    //crop image
    torch_real_projections = torch_real_projections.slice(2, start_dim, start_dim + vae_mask_size);
    torch_real_projections = torch_real_projections.slice(3, start_dim, start_dim + vae_mask_size);
    torch_data = torch_data.slice(2, start_dim, start_dim + vae_mask_size);
    torch_data = torch_data.slice(3, start_dim, start_dim + vae_mask_size);
    torch_real_projections = torch_real_projections.to(device);
    torch_data = torch_data.to(device);
    auto stacked_input = torch::cat({torch_real_projections, torch_data}, 1);
    auto flows = (*(unet_model))->forward(stacked_input);
    auto flow = std::get<0>(flows);
    //warp data to projection
    auto output = torch::nn::functional::grid_sample(torch_data, flow, torch::nn::functional::GridSampleFuncOptions().
            mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
    auto padded_output = torch::nn::functional::pad(output, torch::nn::functional::PadFuncOptions({start_dim, end_pad, start_dim, end_pad}).mode(torch::kConstant));
    wrapped_data.resize(padded_output.numel());
    padded_output = padded_output.to(torch::kCPU);
    std::copy(padded_output.data_ptr<float>(), padded_output.data_ptr<float>() + padded_output.numel(), wrapped_data.begin());
    return 0;
}

int train_unet(std::vector<float> &proj, std::vector<float> &data, std::vector<float>& weight, int part_id, std::vector<float> &wrapped_data){
    //target represents exprimental data, real_projections/proj represent reference projection
    torch::set_num_threads(15);
    (*unet_model)->train();
    torch::Device device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = proj.size()/(vae_image_size*vae_image_size);
    //std::cout << "batch_num: " << batch_num << std::endl;
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    std::vector<long int> w_dims = {batch_num};
    torch::Tensor torch_real_projections = torch::from_blob(proj.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_data = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_weight = torch::from_blob(weight.data(), c10::ArrayRef<long int>(w_dims), options);
    int start_dim = vae_image_size/2 - vae_mask_size/2;
    int end_pad = vae_image_size - (start_dim + vae_mask_size);
    //crop image
    torch_real_projections = torch_real_projections.slice(2, start_dim, start_dim + vae_mask_size);
    torch_real_projections = torch_real_projections.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_real_projections.print();
    torch_data = torch_data.slice(2, start_dim, start_dim + vae_mask_size);
    torch_data = torch_data.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_target.print();
    vae_index++;
    if(vae_index % 4000 == 0){
        //save images
        torch::save(torch_data.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/input"+std::to_string(part_id)+".pt");
        torch::save(torch_real_projections.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/proj"+std::to_string(part_id)+".pt");
        
        //auto input_image = torch_target.index({0});
        //Image<float> debug_img(vae_mask_size, vae_mask_size);
        //memcpy(debug_img.data.data, input_image.data_ptr<float>(), input_image.numel()*sizeof(float));
        //debug_img.write("tmp"+std::to_string(vae_rank)+"/input"+std::to_string(part_id)+".mrc");
        //auto output_image =output.index({0});
        //memcpy(debug_img.data.data, output_image.data_ptr<float>(), output_image.numel()*sizeof(float));
        //debug_img.write("tmp"+std::to_string(vae_rank)+"/output"+std::to_string(part_id)+".mrc");
    }

    //downsample image
    torch_real_projections = torch_real_projections.to(device);
    torch_data = torch_data.to(device);

    //auto torch_proj_mean = torch::mean(torch_real_projections);
    //auto torch_proj_std  = torch::std(torch_real_projections);
    //auto torch_data_mean = torch::mean(torch_data);
    //auto torch_data_std  = torch::std(torch_data);
    //torch_real_projections = (torch_real_projections - torch_proj_mean)/torch_proj_std;
    //torch_data = (torch_data - torch_data_mean)/torch_data_std;

    //auto torch_real_projections_down = torch::nn::functional::interpolate(torch_real_projections, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size/2, vae_mask_size/2})).mode(torch::kBicubic).align_corners(true));
    //auto torch_data_down = torch::nn::functional::interpolate(torch_data, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size/2, vae_mask_size/2})).mode(torch::kBicubic).align_corners(true));
    torch_weight = torch_weight.to(device);
    auto stacked_input = torch::cat({torch_real_projections, torch_data}, 1);
    //auto gpu_input = stacked_input.to(device);
    //auto data_mean = torch::mean(torch_real_projections, {1,2,3}, true, true);
    //auto data_std = std::get<0>(data_std_mean);
    //auto data_mean = std::get<1>(data_std_mean);
    //std::cout << data_std << std::endl;
    //torch_real_projections = (torch_real_projections - data_mean)/(data_std + 1e-4);
    //(*unet_model)->toggleShowSizes();
    //std::cout << "compute flow!" <<std::endl;
    auto flows = (*(unet_model))->forward(stacked_input);
    auto flow = std::get<0>(flows);
    //flow = flow.to(torch::kCPU);
    //upsample flow (no need)
    //flow = torch::nn::functional::interpolate(flow, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size, vae_mask_size})).mode(torch::kBilinear));
    
    //wrap data to projection
    auto output = torch::nn::functional::grid_sample(torch_data, flow, torch::nn::functional::GridSampleFuncOptions().
            mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));

    auto backflow = std::get<1>(flows);
    auto sim_data = torch::nn::functional::grid_sample(torch_real_projections, backflow, torch::nn::functional::GridSampleFuncOptions().
            mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));

    //wrap projection to data
    //compute correlation
    //auto reconstruction_loss = torch::nn::functional::mse_loss(output, torch_real_projections, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
    //auto reconstruction_loss = torch::nn::functional::cosine_similarity(output, torch_target, torch::nn::functional::CosineSimilarityFuncOptions().dim(0));
    //std::cout << "compute cross correlation!" << std::endl;
    torch::Tensor cc;
    torch::Tensor tv_x;
    torch::Tensor tv_y;
    auto compute_cc = [&](auto& x, auto& y){
        auto sum_filt = torch::ones({1, 1, 9, 9}).to(device);
        auto I2 = x*x;
        auto J2 = y*y;
        auto IJ = x*y;
        auto I_sum = torch::nn::functional::conv2d(x, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J_sum = torch::nn::functional::conv2d(y, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I2_sum = torch::nn::functional::conv2d(I2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J2_sum = torch::nn::functional::conv2d(J2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto IJ_sum = torch::nn::functional::conv2d(IJ, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        //auto uI = I_sum / 81.;
        //auto uJ = J_sum / 81.;
        auto cross = IJ_sum - I_sum * J_sum / 81.; //uJ * I_sum - uI * J_sum + uI * uJ * 81.;
        auto I_var = I2_sum - I_sum * I_sum / 81. + 3e-4; //2 * uI * I_sum + uI * uI * 81.;
        auto J_var = J2_sum - J_sum * J_sum / 81. + 3e-4; //2 * uJ * J_sum + uJ * uJ * 81.;
        return cross / torch::sqrt(I_var * J_var);
    };
    cc = compute_cc(torch_real_projections, output);
    cc += compute_cc(torch_data, sim_data);
    //ssim loss
    if(false){
        auto d_win = torch::arange(-4, 5, torch::dtype(torch::kFloat32));
        d_win *= d_win;
        d_win = torch::exp(-d_win / 9.);
        d_win /= torch::sum(d_win);
        d_win = d_win.unsqueeze(1);
        auto sum_filt = d_win.mm(d_win.t()).unsqueeze(0).unsqueeze(0).to(device);

        //auto sum_filt = torch::ones({1, 1, 9, 9}).to(device);
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = output * output;
        auto IJ = torch_real_projections*output;
        auto I_sum = torch::nn::functional::conv2d(torch_real_projections, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J_sum = torch::nn::functional::conv2d(output, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I2_sum = torch::nn::functional::conv2d(I2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J2_sum = torch::nn::functional::conv2d(J2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto IJ_sum = torch::nn::functional::conv2d(IJ, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I_sigma2 = I2_sum - I_sum*I_sum;
        auto J_sigma2 = J2_sum - J_sum*J_sum;
        auto IJ_sigma = IJ_sum - I_sum*J_sum;
        //auto uI = I_sum / 81.;
        //auto uJ = J_sum / 81.;
        //float c1 = 1e-4;
        float c2 = 1e-4;
        auto v1 = IJ_sigma + c2;
        auto v2 = I_sigma2*J_sigma2 + c2;
        //cc = ((2. * I_sum * J_sum + c1) * v1) / ((I_sum*I_sum + J_sum*J_sum + c1) * v2);
        cc = v1 / torch::sqrt(v2);
    }
    if(false) {
        auto IJ = torch_real_projections*output;
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = output*output;
        auto I_ = torch::mean(torch_real_projections, {1,2,3}, true);
        auto J_ = torch::mean(output, {1,2,3}, true);
        auto I2_ = torch::mean(I2, {1,2,3}, true);
        auto J2_ = torch::mean(J2, {1,2,3}, true);
        auto cross = IJ/float(vae_mask_size*vae_mask_size) - I_*J_;
        auto I_var = I2_ - I_*I_;
        auto J_var = J2_ - J_*J_;
        cc = cross / torch::sqrt(I_var*J_var + 1e-5);
    }
    //cc *= torch_weight;
    if(true){
        auto tmp_flow = std::get<2>(flows);
        //tmp_flow.print();
        tv_x = tmp_flow.slice(2, 0, vae_mask_size - 1) - tmp_flow.slice(2, 1, vae_mask_size);
        //dx.print();
        tv_y = tmp_flow.slice(3, 0, vae_mask_size - 1) - tmp_flow.slice(3, 1, vae_mask_size);

        tv_x = torch::nn::functional::pad(tv_x, torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).mode(torch::kConstant));
        tv_y = torch::nn::functional::pad(tv_y, torch::nn::functional::PadFuncOptions({0, 1}).mode(torch::kConstant));
    }

    auto reconstruction_loss = -torch::mean(torch::mean(cc, {1,2,3})*torch_weight);
    //auto tv_loss = 0.04*torch::mean((torch::mean(torch::abs(tv_x), {1,2,3}) + 
    //                                torch::mean(torch::abs(tv_y), {1,2,3}))
    //                                );
    auto tv_loss = 0.04*torch::mean(torch::sqrt(tv_x*tv_x + tv_y*tv_y + 1e-8));
    auto l1_loss = 0.04*torch::mean(torch::abs(std::get<2>(flows)));
    auto loss = reconstruction_loss + (tv_loss + l1_loss)*torch::mean(torch_weight);
    //std::cout << "computed loss!" << std::endl;
    //padding output wrapped image
    auto padded_output = torch::nn::functional::pad(output, torch::nn::functional::PadFuncOptions({start_dim, end_pad, start_dim, end_pad}).mode(torch::kConstant));
    //padded_output.print();
    if(vae_index == 0) vae_optimizer->zero_grad();
    loss.backward();
    if(vae_index != 0 && vae_index %20 == 0) {
        vae_optimizer->step();
        vae_optimizer->zero_grad();
    }
    //vae_index++;
    //save some reconstructions here
    if(vae_index % 4000 == 0){
        //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
        std::cout << "rank: " << vae_rank << " " << vae_index << "/"
            << "Reconstruction loss: "
            << reconstruction_loss.item<double>() / torch::mean(torch_weight)
            << " TV loss: " << tv_loss.item<double>()
            << " weights: " << torch_weight
            << std::endl;
    }
    if(vae_index % 20000 == 0){
        //save model
        std::string model_path = "model" + std::to_string(vae_rank) + ".pt";
        torch::save(*unet_model, model_path);
        std::string optimizer_path = "optimizer" + std::to_string(vae_rank) + ".pt";
        torch::save(*vae_optimizer, optimizer_path);
    }
    if(vae_index % 4000 == 0) {
        torch::save(output.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/output"+std::to_string(part_id)+".pt");
        torch::save(std::get<2>(flows).to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/flow"+std::to_string(part_id)+".pt");
    }
        //convert tensor to vector
    //first multiply with its conditional probability
    //torch_weight = torch_weight.reshape({torch_weight.numel(), 1, 1, 1});
    //padded_output *= torch_weight;
    wrapped_data.resize(padded_output.numel());
    padded_output = padded_output.to(torch::kCPU);
    std::copy(padded_output.data_ptr<float>(), padded_output.data_ptr<float>() + padded_output.numel(), wrapped_data.begin());
}

int train_denoising_unet(std::vector<float> &proj, std::vector<float> &data, std::vector<float>& weight, int part_id, std::vector<float> &wrapped_data){
    //target represents exprimental data, real_projections/proj represent reference projection
    torch::set_num_threads(15);
    (*unet_model)->train();
    torch::Device device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = proj.size()/(vae_image_size*vae_image_size);
    //std::cout << "batch_num: " << batch_num << std::endl;
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    std::vector<long int> w_dims = {batch_num};
    torch::Tensor torch_real_projections = torch::from_blob(proj.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_data = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_weight = torch::from_blob(weight.data(), c10::ArrayRef<long int>(w_dims), options);
    int start_dim = vae_image_size/2 - vae_mask_size/2;
    int end_pad = vae_image_size - (start_dim + vae_mask_size);
    //crop image
    torch_real_projections = torch_real_projections.slice(2, start_dim, start_dim + vae_mask_size);
    torch_real_projections = torch_real_projections.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_real_projections.print();
    torch_data = torch_data.slice(2, start_dim, start_dim + vae_mask_size);
    torch_data = torch_data.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_target.print();
    vae_index++;
    if(vae_index % 4000 == 0){
        //save images
        torch::save(torch_data.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/input"+std::to_string(part_id)+".pt");
        torch::save(torch_real_projections.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/proj"+std::to_string(part_id)+".pt");
    }

    torch_real_projections = torch_real_projections.to(device);
    torch_data = torch_data.to(device);

    //auto torch_proj_mean = torch::mean(torch_real_projections);
    //auto torch_proj_std  = torch::std(torch_real_projections);
    //auto torch_data_mean = torch::mean(torch_data);
    //auto torch_data_std  = torch::std(torch_data);
    //torch_real_projections = (torch_real_projections - torch_proj_mean)/torch_proj_std;
    //torch_data = (torch_data - torch_data_mean)/torch_data_std;

    torch_weight = torch_weight.to(device);
    auto unet_outputs = (*(unet_model))->forward(torch_data);
    auto reconstruction = std::get<0>(unet_outputs);

    //compute correlation
    auto reconstruction_loss = torch::mean(torch::mean((reconstruction - torch_data)*(reconstruction - torch_data), {1,2,3})
                                            *torch_weight)*0.2;
    //std::cout << "compute cross correlation!" << std::endl;
    torch::Tensor cc;
    if(true){
        auto sum_filt = torch::ones({1, 1, 9, 9}).to(device);
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = reconstruction * reconstruction;
        auto IJ = torch_real_projections*reconstruction;
        auto I_sum = torch::nn::functional::conv2d(torch_real_projections, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J_sum = torch::nn::functional::conv2d(reconstruction, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I2_sum = torch::nn::functional::conv2d(I2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J2_sum = torch::nn::functional::conv2d(J2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto IJ_sum = torch::nn::functional::conv2d(IJ, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        //auto uI = I_sum / 81.;
        //auto uJ = J_sum / 81.;
        auto cross = IJ_sum - I_sum * J_sum / 81.; //uJ * I_sum - uI * J_sum + uI * uJ * 81.;
        auto I_var = I2_sum - I_sum * I_sum / 81. + 3e-4; //2 * uI * I_sum + uI * uI * 81.;
        auto J_var = J2_sum - J_sum * J_sum / 81. + 3e-4; //2 * uJ * J_sum + uJ * uJ * 81.;
        cc = cross / torch::sqrt(I_var * J_var);
    }
    if(false) {
        auto IJ = torch_real_projections*reconstruction;
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = reconstruction*reconstruction;
        auto I_ = torch::mean(torch_real_projections, {1,2,3}, true);
        auto J_ = torch::mean(reconstruction, {1,2,3}, true);
        auto I2_ = torch::mean(I2, {1,2,3}, true);
        auto J2_ = torch::mean(J2, {1,2,3}, true);
        auto cross = IJ/float(vae_mask_size*vae_mask_size) - I_*J_;
        auto I_var = I2_ - I_*I_;
        auto J_var = J2_ - J_*J_;
        cc = cross / torch::sqrt(I_var*J_var + 1e-5);
    }
    //cc *= torch_weight;

    auto reconstruction_loss1 = -torch::mean(torch::mean(cc, {1,2,3})*torch_weight);
    auto loss = reconstruction_loss + reconstruction_loss1;
    //std::cout << "computed loss!" << std::endl;
    //padding output wrapped image
    auto padded_output = torch::nn::functional::pad(reconstruction, torch::nn::functional::PadFuncOptions({start_dim, end_pad, start_dim, end_pad}).mode(torch::kConstant));
    //padded_output.print();
    if(vae_index == 0) vae_optimizer->zero_grad();
    loss.backward();
    if(vae_index != 0 && vae_index %20 == 0) {
        vae_optimizer->step();
        vae_optimizer->zero_grad();
    }
    //vae_index++;
    //save some reconstructions here
    if(vae_index % 4000 == 0){
        //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
        std::cout << "rank: " << vae_rank << " " << vae_index << "/"
            << "Reconstruction loss: "
            << reconstruction_loss.item<double>() / torch::mean(torch_weight)
            << " denoising loss: "
            << reconstruction_loss1.item<double>() / torch::mean(torch_weight)
            << " weights: " << torch_weight
            << std::endl;
    }
    if(vae_index % 20000 == 0){
        //save model
        std::string model_path = "model" + std::to_string(vae_rank) + ".pt";
        torch::save(*unet_model, model_path);
        std::string optimizer_path = "optimizer" + std::to_string(vae_rank) + ".pt";
        torch::save(*vae_optimizer, optimizer_path);
    }
    if(vae_index % 4000 == 0) {
        torch::save(reconstruction.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/output"+std::to_string(part_id)+".pt");
    }
        //convert tensor to vector
    //first multiply with its conditional probability
    //torch_weight = torch_weight.reshape({torch_weight.numel(), 1, 1, 1});
    //padded_output *= torch_weight;
    wrapped_data.resize(padded_output.numel());
    padded_output = padded_output.to(torch::kCPU);
    std::copy(padded_output.data_ptr<float>(), padded_output.data_ptr<float>() + padded_output.numel(), wrapped_data.begin());
}

int train_ctf_unet(std::vector<float> &proj, std::vector<float> &data, std::vector<float>& weight, int part_id, std::vector<float> &wrapped_data, 
                   std::vector<float> &Ks, float Q0){
    //target represents exprimental data, real_projections/proj represent reference projection
    torch::set_num_threads(15);
    (*unet_model)->train();
    torch::Device device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int batch_num = proj.size()/(vae_image_size*vae_image_size);
    //std::cout << "batch_num: " << batch_num << std::endl;
    std::vector<long int> dimensions = {batch_num, 1, vae_image_size, vae_image_size};
    std::vector<long int> w_dims = {batch_num};
    torch::Tensor torch_real_projections = torch::from_blob(proj.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_data = torch::from_blob(data.data(), c10::ArrayRef<long int>(dimensions), options);
    torch::Tensor torch_weight = torch::from_blob(weight.data(), c10::ArrayRef<long int>(w_dims), options);
    int start_dim = vae_image_size/2 - vae_mask_size/2;
    int end_pad = vae_image_size - (start_dim + vae_mask_size);
    //crop image
    torch_real_projections = torch_real_projections.slice(2, start_dim, start_dim + vae_mask_size);
    torch_real_projections = torch_real_projections.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_real_projections.print();
    torch_data = torch_data.slice(2, start_dim, start_dim + vae_mask_size);
    torch_data = torch_data.slice(3, start_dim, start_dim + vae_mask_size);
    //torch_target.print();
    vae_index++;
    if(vae_index % 4000 == 0){
        //save images
        torch::save(torch_data.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/input"+std::to_string(part_id)+".pt");
        torch::save(torch_real_projections.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/proj"+std::to_string(part_id)+".pt");
    }

    //downsample image
    torch_real_projections = torch_real_projections.to(device);
    torch_data = torch_data.to(device);

    //auto torch_proj_mean = torch::mean(torch_real_projections);
    //auto torch_proj_std  = torch::std(torch_real_projections);
    //auto torch_data_mean = torch::mean(torch_data);
    //auto torch_data_std  = torch::std(torch_data);
    //torch_real_projections = (torch_real_projections - torch_proj_mean)/torch_proj_std;
    //torch_data = (torch_data - torch_data_mean)/torch_data_std;

    //auto torch_real_projections_down = torch::nn::functional::interpolate(torch_real_projections, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size/2, vae_mask_size/2})).mode(torch::kBicubic).align_corners(true));
    //auto torch_data_down = torch::nn::functional::interpolate(torch_data, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size/2, vae_mask_size/2})).mode(torch::kBicubic).align_corners(true));
    torch_weight = torch_weight.to(device);
    auto stacked_input = torch::cat({torch_real_projections, torch_data}, 1);
    //std::cout << "compute flow!" <<std::endl;
    auto flows = (*(unet_model))->forward(stacked_input);
    auto flow = std::get<0>(flows);
    //flow = flow.to(torch::kCPU);
    //upsample flow (no need)
    //flow = torch::nn::functional::interpolate(flow, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({vae_mask_size, vae_mask_size})).mode(torch::kBilinear));
    
    //wrap data to projection
    auto output = torch::nn::functional::grid_sample(torch_data, flow, torch::nn::functional::GridSampleFuncOptions().
            mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));

    auto backflow = std::get<1>(flows);
    auto sim_data = torch::nn::functional::grid_sample(torch_real_projections, backflow, torch::nn::functional::GridSampleFuncOptions().
            mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));

    //wrap projection to data
    //compute correlation
    //auto reconstruction_loss = torch::nn::functional::mse_loss(output, torch_real_projections, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
    //auto reconstruction_loss = torch::nn::functional::cosine_similarity(output, torch_target, torch::nn::functional::CosineSimilarityFuncOptions().dim(0));
    //std::cout << "compute cross correlation!" << std::endl;
    torch::Tensor cc;
    torch::Tensor tv_x;
    torch::Tensor tv_y;
    auto compute_cc = [&](auto& x, auto& y){
        auto sum_filt = torch::ones({1, 1, 9, 9}).to(device);
        auto I2 = x*x;
        auto J2 = y*y;
        auto IJ = x*y;
        auto I_sum = torch::nn::functional::conv2d(x, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J_sum = torch::nn::functional::conv2d(y, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I2_sum = torch::nn::functional::conv2d(I2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J2_sum = torch::nn::functional::conv2d(J2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto IJ_sum = torch::nn::functional::conv2d(IJ, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        //auto uI = I_sum / 81.;
        //auto uJ = J_sum / 81.;
        auto cross = IJ_sum - I_sum * J_sum / 81.; //uJ * I_sum - uI * J_sum + uI * uJ * 81.;
        auto I_var = I2_sum - I_sum * I_sum / 81. + 3e-4; //2 * uI * I_sum + uI * uI * 81.;
        auto J_var = J2_sum - J_sum * J_sum / 81. + 3e-4; //2 * uJ * J_sum + uJ * uJ * 81.;
        return cross / torch::sqrt(I_var * J_var);
    };
    cc = compute_cc(torch_real_projections, output);
    cc += compute_cc(torch_data, sim_data);
    //ssim loss
    if(false){
        auto d_win = torch::arange(-4, 5, torch::dtype(torch::kFloat32));
        d_win *= d_win;
        d_win = torch::exp(-d_win / 9.);
        d_win /= torch::sum(d_win);
        d_win = d_win.unsqueeze(1);
        auto sum_filt = d_win.mm(d_win.t()).unsqueeze(0).unsqueeze(0).to(device);

        //auto sum_filt = torch::ones({1, 1, 9, 9}).to(device);
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = output * output;
        auto IJ = torch_real_projections*output;
        auto I_sum = torch::nn::functional::conv2d(torch_real_projections, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J_sum = torch::nn::functional::conv2d(output, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I2_sum = torch::nn::functional::conv2d(I2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto J2_sum = torch::nn::functional::conv2d(J2, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto IJ_sum = torch::nn::functional::conv2d(IJ, sum_filt, torch::nn::functional::Conv2dFuncOptions().stride(1));
        auto I_sigma2 = I2_sum - I_sum*I_sum;
        auto J_sigma2 = J2_sum - J_sum*J_sum;
        auto IJ_sigma = IJ_sum - I_sum*J_sum;
        //auto uI = I_sum / 81.;
        //auto uJ = J_sum / 81.;
        //float c1 = 1e-4;
        float c2 = 1e-4;
        auto v1 = IJ_sigma + c2;
        auto v2 = I_sigma2*J_sigma2 + c2;
        //cc = ((2. * I_sum * J_sum + c1) * v1) / ((I_sum*I_sum + J_sum*J_sum + c1) * v2);
        cc = v1 / torch::sqrt(v2);
    }
    if(false) {
        auto IJ = torch_real_projections*output;
        auto I2 = torch_real_projections*torch_real_projections;
        auto J2 = output*output;
        auto I_ = torch::mean(torch_real_projections, {1,2,3}, true);
        auto J_ = torch::mean(output, {1,2,3}, true);
        auto I2_ = torch::mean(I2, {1,2,3}, true);
        auto J2_ = torch::mean(J2, {1,2,3}, true);
        auto cross = IJ/float(vae_mask_size*vae_mask_size) - I_*J_;
        auto I_var = I2_ - I_*I_;
        auto J_var = J2_ - J_*J_;
        cc = cross / torch::sqrt(I_var*J_var + 1e-5);
    }
    //cc *= torch_weight;
    if(true){
        auto tmp_flow = std::get<2>(flows);
        //tmp_flow.print();
        tv_x = tmp_flow.slice(2, 0, vae_mask_size - 1) - tmp_flow.slice(2, 1, vae_mask_size);
        //dx.print();
        tv_y = tmp_flow.slice(3, 0, vae_mask_size - 1) - tmp_flow.slice(3, 1, vae_mask_size);

        tv_x = torch::nn::functional::pad(tv_x, torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).mode(torch::kConstant));
        tv_y = torch::nn::functional::pad(tv_y, torch::nn::functional::PadFuncOptions({0, 1}).mode(torch::kConstant));
    }

    auto reconstruction_loss = -torch::mean(torch::mean(cc, {1,2,3})*torch_weight);
    //auto tv_loss = 0.04*torch::mean((torch::mean(torch::abs(tv_x), {1,2,3}) + 
    //                                torch::mean(torch::abs(tv_y), {1,2,3}))
    //                                );
    auto tv_loss = 0.04*torch::mean(torch::sqrt(tv_x*tv_x + tv_y*tv_y + 1e-8));
    auto l1_loss = 0.04*torch::mean(torch::abs(std::get<2>(flows)));
    auto loss = reconstruction_loss + (tv_loss + l1_loss)*torch::mean(torch_weight);
    //std::cout << "computed loss!" << std::endl;
    //padding output wrapped image
    auto padded_output = torch::nn::functional::pad(output, torch::nn::functional::PadFuncOptions({start_dim, end_pad, start_dim, end_pad}).mode(torch::kConstant));
    //padded_output.print();
    if(vae_index == 0) vae_optimizer->zero_grad();
    loss.backward();
    if(vae_index != 0 && vae_index %20 == 0) {
        vae_optimizer->step();
        vae_optimizer->zero_grad();
    }
    //vae_index++;
    //save some reconstructions here
    if(vae_index % 4000 == 0){
        //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
        std::cout << "rank: " << vae_rank << " " << vae_index << "/"
            << "Reconstruction loss: "
            << reconstruction_loss.item<double>() / torch::mean(torch_weight)
            << " TV loss: " << tv_loss.item<double>()
            << " weights: " << torch_weight
            << std::endl;
    }
    if(vae_index % 20000 == 0){
        //save model
        std::string model_path = "model" + std::to_string(vae_rank) + ".pt";
        torch::save(*unet_model, model_path);
        std::string optimizer_path = "optimizer" + std::to_string(vae_rank) + ".pt";
        torch::save(*vae_optimizer, optimizer_path);
    }
    if(vae_index % 4000 == 0) {
        torch::save(output.to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/output"+std::to_string(part_id)+".pt");
        torch::save(std::get<2>(flows).to(torch::kCPU), "tmp"+std::to_string(vae_rank)+"/flow"+std::to_string(part_id)+".pt");
    }
        //convert tensor to vector
    //first multiply with its conditional probability
    //torch_weight = torch_weight.reshape({torch_weight.numel(), 1, 1, 1});
    //padded_output *= torch_weight;
    wrapped_data.resize(padded_output.numel());
    padded_output = padded_output.to(torch::kCPU);
    std::copy(padded_output.data_ptr<float>(), padded_output.data_ptr<float>() + padded_output.numel(), wrapped_data.begin());
}
