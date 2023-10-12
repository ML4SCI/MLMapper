# Deriving planetary surface composition from orbiting observations from spacecraft

<div align="center">


A Google Summer of Code 2023 Project Repository.<br>
The code was written by Sandeepan Dhoundiyal during the Google Summer of Code (GSoC) 2023 for the Lunar Prospector project which is a part of the ML4SCI umbrella organization. It was performed under the supervision of Patrick Peplowski (JHUAPL), Mauricio Ayllon Unzueta (NASA Goddard Space Flight Center / Catholic University of America) and Jack Wilson (JHUAPL).</b>

  <a href="https://ml4sci.org/" target="_blank"><img alt="gsoc@ml4sci" height="200px" src="https://raw.githubusercontent.com/eraraya-ricardo/GSoC-QCNN/main/assets/gsoc%40ml4sci.jpeg" /></a>
    
</div>

This README describes the preprocessed and curated <a href="https://github.com/sdhoundiyal/MLMapper/tree/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset" target="_blank">dataset</a> used to train the unmixing model described in the <a href="https://github.com/ML4SCI/MLMapper/tree/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset" target="_blank">accompanying blogpost</a>, as well as the <a href="https://github.com/ML4SCI/MLMapper/tree/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Notebooks" target="_blank">code</a> used to train it.


## Dataset

The dataset consists of Gamma-ray spectra collected by the Lunar Prospector's (LP) Gamma Ray Spectrometer (GRS) for 5°×5° sections of the Lunar Surface along with abundances of ten elements. Abundances are in Wt% for seven of the ten elements (Al, Ca, Fe, Mg, O, Si, Ti) and in PPM for the others (K, Th, U). The spectra underwent multiple levels of pre-processing, therefore spectra at each level of pre-preprocessing are included in the repository as .npy files (arrays saved/read using Numpy). Each step and the file containing the output from the step (and units) are listed below (check out the blog for details on pre-processing).

1. Lower Latitude demarcating each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/minLatitudePerSection.npy">datasets/minLatitudePerSection.npy</a> (°).
2. Upper Latitude demarcating each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/maxLatitudePerSection.npy">datasets/maxLatitudePerSection.npy</a> (°).
3. Eastern most Latitude demarcating each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/maxLongitudePerSection.npy">datasets/maxLongitudePerSection.npy</a> (°).
4. Western most Latitude demarcating each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/minLongitudePerSection.npy">datasets/minLongitudePerSection.npy</a> (°).
5. Number of spectra in each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/noOfSpectraPerSection.npy">datasets/noOfSpectraPerSection.npy</a> (unitless).
6. Total time the sensor was active in each 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/totalActiveTimePerSection.npy">datasets/totalActiveTimePerSection.npy</a> (minutes).
7. Sum of all spectra collected in 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/summedSpectraPerSection.npy">datasets/summedSpectraPerSection.npy</a> (Counts).
8. Normalized spectrum for each in 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/normalizedSpectraPerSection.npy">datasets/normalizedSpectraPerSection.npy</a> (Counts/minute).
9. Log-scaled Normalized spectrum for each in 5°×5° section - <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/logScaledNormalizedSpectraPerSection.npy">datasets/logScaledNormalizedSpectraPerSection.npy</a> (log(Counts/minute)).

The afformentioned data as well as those at subsequent levels of preprocessing are included in a <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Dataset/GRSFiveDegreeSectionDataset.pkl">pickled Pandas dataframe </a>. Preprocessing levels not mentioned above are described below with a brief description of the output as well as their title in the dataframe (and units).

1. The log scaled spectra denosied using a 31-channel wide Savitsky-Golay filter - Denoised Log Scaled Spectra (log(Counts/minute))
2. The denoised spectra with the continuum removed - Continuum Removed Denoised Log Scaled Spectra (Ratioed log(Counts/minute))
3. The Continuum removed spectra converted to unit vectors by dividing them with their respective L2-norm - Normalized Continuum Removed Denoised Log Scaled Spectra (Ratioed log(Counts/minute))

## Code

The core of the code is organized into two notebooks, each used to train one version of the model. Both models are autoencoders where the encoder consists of two parallel netoworks made from Residual blocks combined with the Convolutional Block Attention Module (CBAM) and estimate the abundances of the ten elements. The decoder in each learns to reconstruct the input (pre-processed) spectrum using the elemental abundances. Details on the two models, their loss functions, and regularizations, are provided in the blog.

# 1. Linear decoder
The first version of the model consits of a linar decoder which reconstructs the input spectrum by weighing each element's spectrum (elemental spectra are learnt from the data) by its respective abunadance and adding them together. The notebook titled is <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Notebooks/LinearDecoder/CBAM_based_Encoder_With_Linear_Decoder_And_Cosine_Penalty_OverSubset.ipynb">CBAM_based_Encoder_With_Linear_Decoder_And_Cosine_Penalty_OverSubset.ipynb</a> was used to train the model over a subset of the dataset. The Subset is created by identifying the spectra most unlike the central tendency of the dataset, and each other, using the Discovery through Eigen Modelling of Uninteresting Data (DEMUD) algorithm. While the notebook titled <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Notebooks/LinearDecoder/CBAM_based_Encoder_With_Linear_Decoder_And_Cosine_Penalty_FullDataset.ipynb">CBAM_based_Encoder_With_Linear_Decoder_And_Cosine_Penalty_FullDataset.ipynb</a> used used to train the model over the entire dataset cross-validated across 5-folds.

# 2. Post-Non-linear decoder
The second version, adds a series of residual layers after the liner decoder to create a more faithful recontruction. It was trained using the notebook titled:  <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Notebooks/postNonLinearDecoder/SSIM_Loss_With_Firing_Strength.ipynb">postNonLinearDecoder/SSIM_Loss_With_Firing_Strength.ipynb</a>. Due to time contrained, it could only be trained over one fold, i.e. 80% of the dataset and tested over the remaining 20%.



Finally, additional code was written to generate an augmented dataset (by varying the size of the Savitsky-Golay filter used to denoise the spectra). It is in the notebook titled <a href="https://github.com/ML4SCI/MLMapper/blob/main/Lunar_Prospector/Deriving_Surface_Composition_Sandeepan_Dhoundiyal/Final_Spectral_unmixing/Notebooks/DataAugmentation/generateAugmentedDataset.ipynb">DataAugmentation/generateAugmentedDataset.ipynb</a>. However, due to time constraints, the network could not be trained over this augemented dataset.





