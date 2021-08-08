#include "src/vae/api.h"
#include "src/image.h"
#include "src/multidim_array.h"

int main(int argc, char *argv[])
{
    FileName mrc_file = argv[1];
    Image<RFLOAT> Im; 
    Image<RFLOAT> neural_Im;
    //read in mrc
    Im.read(mrc_file);

    std::vector<float> vol(MULTIDIM_SIZE(Im()));
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Im())
    {
        vol[n] = DIRECT_MULTIDIM_ELEM(Im(), n);
    }

    //call the neural representation
    std::vector<float> neural_vol;

    neural_volume_reconstruction(vol, XSIZE(Im()), neural_vol);

    //save the volume reconstructed by neural network
    neural_Im().resize(Im());
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(neural_Im())
    {
        DIRECT_MULTIDIM_ELEM(neural_Im(), n) = neural_vol[n];
    }

    FileName neural_file = "neural_" + mrc_file;

    neural_Im().write(neural_file);
}
