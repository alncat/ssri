//this file implement the neural volumetric respresentation
#pragma once
#include <torch/torch.h>
#include <utility>
#include <math.h>

struct NeuralVolumeImpl : public torch::nn::Module 
{
    public:
    torch::Tensor b_vec;
    int embedding_size;
    int total_depth;
    int volume_size;
    double sigma;
    std::deque<torch::nn::Sequential> mlps;
    torch::nn::Linear output{nullptr};
    //torch::Device device{nullptr};

    torch::Tensor positional_encoding(torch::Tensor& coords) 
    {
        auto sin_enc = torch::sin(torch::matmul(coords*M_PI*2., b_vec));
        auto cos_enc = torch::cos(torch::matmul(coords*M_PI*2., b_vec));
        auto encs    = torch::cat({sin_enc, cos_enc}, -1);
        return encs;
    }

    torch::nn::Sequential construct_mlp(int in_channels, int out_channels) 
    {
        return torch::nn::Sequential(
                torch::nn::Linear(in_channels, out_channels),
                torch::nn::ReLU());
    }

    NeuralVolumeImpl(int embedding_size_, int total_depth_, int volume_size_, double sigma_, torch::Device device_) 
    {
        embedding_size = embedding_size_;
        total_depth = total_depth_;
        volume_size = volume_size_;
        sigma = sigma_;
        //device = device_;
        mlps.push_front(construct_mlp(embedding_size*2, embedding_size));
        register_module("mlp" + std::to_string(0), mlps.front());
        for(int i = 1; i < total_depth - 1; i++)
        {
            mlps.push_front(construct_mlp(embedding_size, embedding_size));
            register_module("mlp" + std::to_string(i), mlps.front());
        }
        //final regression layer
        output = torch::nn::Linear(torch::nn::LinearOptions(embedding_size, 1).bias(false));
        register_module("output", output);
        std::cout << "constructed" << std::endl;
        //sampling b vectors from gaussian
        b_vec = torch::normal(0., sigma, {3, embedding_size}).to(device_);
    }

    torch::Tensor forward(torch::Tensor& coords) 
    {
        //apply positional encoding
        auto encoded_input = positional_encoding(coords);
        std::cout << "encoded: " << encoded_input.sizes() << std::endl;

        std::vector<torch::Tensor> embedding_outputs;

        embedding_outputs.push_back(mlps[mlps.size() - 1]->forward(encoded_input));

        for(int i = mlps.size() - 2; i >= 0; i--)
        {
            embedding_outputs.push_back(mlps[i]->forward(embedding_outputs.back()));
        }
        //output tensor is a vectorized volume, need to reshape
        auto vol_out =  output->forward(embedding_outputs.back());
        vol_out = vol_out.flatten();
        return vol_out;
    }

    //construct a grid and reshape it to a vector
    torch::Tensor grid_coords()
    {
        auto x = torch::linspace(0, 1, volume_size);
        auto grids = torch::meshgrid({x, x, x});
        auto coords = torch::stack(grids, -1);
        coords = coords.view({volume_size*volume_size*volume_size, 3});
        return coords;
    }
};

TORCH_MODULE_IMPL(NeuralVolume, NeuralVolumeImpl);
