#include "nvol.h"

void neural_volume_reconstruction(std::vector<float>& volume, int vol_size, std::vector<float>& reconstructed_vol)
{
    torch::Device device(torch::kCUDA, 0);

    //reconstruct a volume using neural network
    //convert 3d volume to torch tensor
    std::vector<long int> dimensions = {vol_size, vol_size, vol_size};
    torch::Tensor torch_volume = torch::from_blob(volume.data(), 
                                        c10::ArrayRef<long int>(dimensions), 
                                        torch::TensorOptions().dtype(torch::kFloat32));
    torch_volume = torch_volume.view(-1).to(device);
    //construct a neural representation
    std::cout << "torch_vol: " << torch_volume.sizes() << std::endl;
    NeuralVolume nvol_model(256, 16, vol_size, 1., device);
    nvol_model->to(device);

    int batch_size = vol_size*vol_size/4;
    int batch_num  = vol_size*vol_size*vol_size / batch_size;
    //fitting neuralvolume

    auto compute_loss = [&](auto& coords, int j){
        auto mlp_vol = nvol_model->forward(coords);
        //mlp_vol = mlp_vol.view(dimensions);
        //compare it with respect to experimental vol
        auto vol_sel = torch_volume.index({torch::indexing::Slice(j*batch_size, (j+1)*batch_size)});
        auto loss = torch::nn::functional::mse_loss(mlp_vol, vol_sel);
        return loss;
    };

    //construct an optimizer
    torch::optim::AdamW optimizer(nvol_model->parameters());

    nvol_model->train();

    auto all_coords = nvol_model->grid_coords().to(device);
    //iterate for several steps
    for(int i = 0; i < 500; i++)
    {
        for(int j = 0; j < batch_num; j++)
        {
            optimizer.zero_grad();
            auto coords = all_coords.index({torch::indexing::Slice(j*batch_size, (j+1)*batch_size)});
            auto loss = compute_loss(coords, j);
            std::cout << loss.item<float>() << std::endl;
            loss.backward();
            optimizer.step();
        }
    }
    std::cout << "optimized" << std::endl;

    //store the reconstruction_vol
    //compute the whole volume after training
    auto coords = all_coords.index({torch::indexing::Slice(0, batch_size)});
    auto final_vol = nvol_model->forward(coords);
    for(int j = 1; j < batch_num; j++)
    {
        coords = all_coords.index({torch::indexing::Slice(j*batch_size, (j+1)*batch_size)});
        auto tmp_vol = nvol_model->forward(coords);
        final_vol = torch::cat({final_vol, tmp_vol}, -1);
    }
    final_vol = final_vol.to(torch::kCPU);
    reconstructed_vol.resize(final_vol.numel());

    std::copy(final_vol.data_ptr<float>(), final_vol.data_ptr<float>() + final_vol.numel(), reconstructed_vol.begin());
}
