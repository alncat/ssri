#include "real_reconstruction.h"

static std::unique_ptr<RealReconstructor> real_reconstructor;
static std::unique_ptr<torch::optim::AdamW> optimizer;
static int image_size;
static int masked_size;
static int training_index = 0;
static int reconstructor_rank;

void initialise_real_reconstructor(int rank, int device, int mask_size, int full_size, bool do_damping, float angpix, float learning_rate)
{
    image_size  = full_size;
    masked_size = mask_size;
    //intialise real volume
    real_reconstructor = std::make_unique<RealReconstructor>(full_size, do_damping, angpix, device);
    reconstructor_rank = rank;
    std::vector<torch::Tensor> params;
    params.push_back((*real_reconstructor)->volume);
    optimizer = std::make_unique<torch::optim::AdamW>(params, learning_rate);
}

inline torch::Tensor crop_fft_to_fit(torch::Tensor& x, int current_size, int full_size)
{
    //auto xas1 = torch::slice(x, x.dim() - 1, 0, current_size/2 + 1);
    //xas1 = torch::slice(xas1, x.dim(), 0, current_size/2 + 1);
    using namespace torch::indexing;
    int x_dim = current_size/2 + 1;
    auto xas1 = x.index({Slice(), Slice(), Slice(None, x_dim), Slice(None, x_dim)});
    auto xas2 = x.index({Slice(), Slice(), Slice(-current_size/2 + 1 + full_size, full_size), Slice(None, x_dim)});
    //auto xas2 = torch::slice(x, x.dim() - 1, full_size - current_size/2 + 1, full_size);
    //xas2 = torch::slice(xas2, x.dim(), 0, current_size/2 + 1);
    auto xa_win = torch::cat({xas1, xas2}, 3);
    return xa_win;
}

inline torch::Tensor expand_fft_to_full(torch::Tensor& x, int current_size, int full_size)
{
    using namespace torch::indexing;
    int x_dim = current_size/2 + 1;
    auto x_full = torch::zeros({x.size(0), x.size(1), full_size, full_size/2 + 1, 2}, torch::dtype(torch::kFloat32));
    //convert to complex array
    x_full = torch::view_as_complex(x_full);
    auto xs1 = x.index({Slice(), Slice(), Slice(None, x_dim), Slice()});
    x_full.index_put_({Slice(), Slice(), Slice(None, x_dim), Slice(None, x_dim)}, xs1);
    auto xs2 = x.index({Slice(), Slice(), Slice(x_dim, current_size), Slice()});
    x_full.index_put_({Slice(), Slice(), Slice(full_size - current_size/2 + 1, full_size), Slice(None, x_dim)}, xs2);
    return x_full;
}

inline torch::Tensor expand_real_to_full(torch::Tensor& x, int current_size, int full_size)
{
    using namespace torch::indexing;
    int x_dim = current_size/2 + 1;
    auto x_full = torch::zeros({x.size(0), x.size(1), full_size, full_size/2 + 1}, torch::dtype(torch::kFloat32));
    auto xs1 = x.index({Slice(), Slice(), Slice(None, x_dim), Slice()});
    x_full.index_put_({Slice(), Slice(), Slice(None, x_dim), Slice(None, x_dim)}, xs1);
    auto xs2 = x.index({Slice(), Slice(), Slice(x_dim, current_size), Slice()});
    x_full.index_put_({Slice(), Slice(), Slice(full_size - current_size/2 + 1, full_size), Slice(None, x_dim)}, xs2);
    return x_full;
}

void set_real_reconstructor_volume(MultidimArray<float>& vol_, bool logging)
{
    (*real_reconstructor)->set_volume(vol_, logging);
}

void train_real_reconstructor(
        std::vector<float>& images, std::vector<float>& corr, std::vector<float>& ref_projections, 
        std::vector<float>& weight, MultidimArray<float>& ref_ctf, 
        int current_size, std::vector<float>& thetas, 
        std::vector<float>& deltas, float deltaa, std::vector<float>& Ks, float Q0, int part_id)
{
    //convert images to torch tensor
    int x_dim = current_size/2 + 1;
    int shift = current_size*x_dim;
    int batch_num = (images.size())/(current_size*x_dim*2);
    std::vector<long int> image_dim = {batch_num, 1, current_size, x_dim, 2};

    //images are complex data
    torch::Tensor torch_images = torch::from_blob(images.data(), c10::ArrayRef<long int>(image_dim), 
                                                torch::dtype(torch::kFloat32));
    torch_images = torch::view_as_complex(torch_images);
    // reference ctf is already cropped
    
    auto full_images = expand_fft_to_full(torch_images, current_size, image_size);

    // correlation is already cropped
    std::vector<long int> corr_dim = {1, 1, current_size, current_size/2 + 1};
    torch::Tensor torch_corr = torch::from_blob(corr.data(), c10::ArrayRef<long int>(corr_dim),
                                                torch::dtype(torch::kFloat32));
    auto full_corr = expand_real_to_full(torch_corr, current_size, image_size);

    torch::Tensor torch_ctf = torch::from_blob(ref_ctf.data, c10::ArrayRef<long int>(corr_dim),
                                                torch::dtype(torch::kFloat32));
    auto full_ctf = expand_real_to_full(torch_ctf, current_size, image_size);

    std::vector<long int> weight_dim = {batch_num};
    //weight at zero index represents the total weight
    torch::Tensor torch_weight = torch::from_blob(weight.data() + 1, c10::ArrayRef<long int>(weight_dim),
                                                torch::dtype(torch::kFloat32));

    //get fourier transform of images

    //torch_images = torch::roll(torch_images, {image_size/2, image_size/2}, {2, 3});
    //auto fft_images = torch::fft::rfft2(torch_images);
    //auto fft_images_crop = crop_fft_to_fit(fft_images, current_size, image_size);

    //convert euler angles to torch tensor
    std::vector<long int> euler_dim = {batch_num, 3};
    torch::Tensor torch_thetas = torch::from_blob(thetas.data(), c10::ArrayRef<long int>(euler_dim));
    //move arrays to gpu
    torch::Device device(torch::kCUDA, 0);
    full_images = full_images.to(device);
    full_corr   = full_corr.to(device);
    torch_thetas = torch_thetas.to(device);
    torch_weight = torch_weight.to(device);

    if(part_id == 1989)
    {
        std::cout << "shapes: " << images.size() << " " << corr.size() << " " << ref_projections.size() << " " 
            << current_size << " " << weight.size() << " " << torch_thetas.size(0) << std::endl;
    }

    //get real reconstruction by projection
    auto projection = (*real_reconstructor)->forward(torch_thetas);
    //fftshift before fourier transform
    auto projection_shifted = torch::fft::fftshift(projection, std::vector<long int>({2, 3}));
    //projection_shifted.print();
    //compute fourier transform and using forward scale
    auto projection_fft = torch::fft::rfft2(projection_shifted, c10::nullopt, {-2, -1}, "forward");

    //projection_fft.print();
    //compute ctf
    auto ctf = (*real_reconstructor)->compute_ctf(deltas, deltaa, Ks, Q0);

    //slice both
    //auto fft_projection_crop = crop_fft_to_fit(fft_projection, current_size, image_size);
    //auto ctf_crop = crop_fft_to_fit(ctf, current_size, image_size);

    //multiply with ctf
    auto projection_fft_ctf = projection_fft * ctf;
    //projection_fft_ctf.print();

    //construct loss
    auto correlation = torch::real(full_images.conj() * projection_fft_ctf);

    //construct projection norm
    auto proj_norm = torch::real(projection_fft_ctf.conj() * projection_fft_ctf);

    //apply per shell variance correction

    auto loss = (correlation + proj_norm) * full_corr;

    loss = torch::sum(torch::mean(loss, {1, 2, 3}) * torch_weight);

    //backward pass
    if(training_index == 0)
        optimizer->zero_grad();
    //
    loss.backward();

    if(training_index % 10 == 0 && training_index > 0)
    {
        optimizer->step();
        optimizer->zero_grad();
    }

    training_index++;

    //print loss

    if(training_index % 1000 == 0)
    {
        std::cout << loss.item<float>() << std::endl;
    }

    //save volume as mrc
    if(part_id == 1989)
    {
        //save reference projection
        torch::Tensor torch_projection = torch::from_blob(ref_projections.data(), image_dim, torch::dtype(torch::kFloat32));
        torch_projection = torch::view_as_complex(torch_projection);
        auto full_projection = expand_fft_to_full(torch_projection, current_size, image_size);
        //save reference projection
        torch::save(full_projection.to(torch::kCPU), "tmp"+std::to_string(reconstructor_rank)+"/ref_proj"+std::to_string(part_id)+".pt");

        //save projection generated in real space
        torch::save(projection_fft.to(torch::kCPU), "tmp"+std::to_string(reconstructor_rank)+"/proj"+std::to_string(part_id)+".pt");

        //save images
        torch::save(full_images.to(torch::kCPU), "tmp"+std::to_string(reconstructor_rank)+"/real_imag"+std::to_string(part_id)+".pt");

        //save ctfs
        torch::save(full_ctf, "tmp"+std::to_string(reconstructor_rank)+"/ref_ctf"+std::to_string(part_id)+".pt");
        torch::save(ctf.to(torch::kCPU), "tmp"+std::to_string(reconstructor_rank)+"/ctf"+std::to_string(part_id)+".pt");

        //(*real_reconstructor)->save_volume();

        //Image<float> tmp_vol(image_size, image_size, image_size);
        //memcpy(tmp_vol.data.data, (*real_reconstructor)->volume.data_ptr<float>(), (*real_reconstructor)->volume.numel()*sizeof(float));
        //tmp_vol.write("tmp.mrc");
        //save projections and etc
    }
}
