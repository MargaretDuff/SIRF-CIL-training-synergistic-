import os
import numpy as np

import brainweb
import sirf.STIR as pet
msg = pet.MessageRedirector()

from helper_functions import fwhm_to_sigma, psf, make_acquisition_model, add_poission_noise

from cil.optimisation.operators import  BlurringOperator, CompositionOperator

import argparse

parser = argparse.ArgumentParser(description='Generate PET amyloid data')
parser.add_argument('--bw_seed', type=int, default=1337, help='Brainweb seed')
parser.add_argument('--crop_dim', type=int, nargs=3, default=(8,128,128), help='Crop dimensions')
parser.add_argument('--noise_seed', type=int, default=5, help='Noise seed')
parser.add_argument('--noise_level', type=int, default=1, help='Noise level')

args = parser.parse_args()

data_path = os.path.join(os.path.dirname(__file__), 'data')

def main(args):

    fname, url= sorted(brainweb.utils.LINKS.items())[0]
    files = brainweb.get_file(fname, url, data_path)
    data = brainweb.load_file(os.path.join(data_path, fname))

    brainweb.seed(args.bw_seed)

    vol = brainweb.get_mmr_fromfile(os.path.join(data_path, fname),
            petNoise=1, t1Noise=0.75, t2Noise=0.75,
            petSigma=1, t1Sigma=1, t2Sigma=1,
            PetClass=brainweb.Amyloid)
    
    arr_dict = {'PET_amyloid': vol['PET'],'uMap': vol['uMap']}

    crop_dim = (8,128,128)

    for key, image in arr_dict.items():
        shape = image.shape
        z_start, y_start, x_start = (shape[0] - crop_dim[0])//2, (shape[1] - crop_dim[1])//2, (shape[2] - crop_dim[2])//2
        z_end, y_end, x_end = z_start + crop_dim[0], y_start + crop_dim[1], x_start + crop_dim[2]
        
        arr_dict[key] = image[z_start:z_end, y_start:y_end, x_start:x_end]

    image_dict = {}
    vsize = (2.03125, 2.08626, 2.08626) # voxel sizes in mm
    for key, image in arr_dict.items():
        image_dict[key] = pet.ImageData()
        image_dict[key].initialise(dim = crop_dim, vsize = vsize)
        image_dict[key].fill(image)
        if key == 'PET_amyloid':
            image_dict[key].write(os.path.join(data_path, f'{key}_b{args.bw_seed}.hv'))
    
    PSF=psf(15, fwhm=(7,7,7), voxel_size=vsize)
    convolve=BlurringOperator(PSF, image_dict['PET_amyloid'])
    acq_model = make_acquisition_model(pet.AcquisitionData(os.path.join(data_path, 'template_sinogram.hs')), image_dict['PET_amyloid'], image_dict['uMap'])
    blurred_acq_model=CompositionOperator(acq_model, convolve)
    
    blurred_sinogram=blurred_acq_model.direct(image_dict['PET_amyloid'])
    blurred_noisy_sinogram=add_poission_noise(blurred_sinogram, noise_level=args.noise_level, seed=args.noise_seed)
    blurred_noisy_sinogram.write(os.path.join(data_path,f'bw_amyloid_blurred_noisy_sinogram_b_{args.bw_seed}_n{args.noise_seed}.hs'))

    objective_function = pet.make_Poisson_loglikelihood(blurred_noisy_sinogram, acq_model=acq_model)
    objective_function.set_num_subsets(8)
    reconstructor = pet.OSMAPOSLReconstructor()
    reconstructor.set_num_subiterations(8)
    reconstructor.set_objective_function(objective_function)
    reconstructor.set_up(image_dict['PET_amyloid'])
    
    cyl = pet.TruncateToCylinderProcessor()
    cyl.set_strictly_less_than_radius(True)
    
    current_estimate = image_dict['PET_amyloid'].get_uniform_copy(1)
    
    for _ in range(50):
        reconstructor.reconstruct(current_estimate)
        cyl.apply(current_estimate)
        
    current_estimate.write(f'data/OSEM_amyloid_b{args.bw_seed}_n{args.noise_seed}.hv')

if __name__ == '__main__':
    main(parser.parse_args())
    