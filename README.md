# CBS dataset
· Our CBS dataset has now been uploaded to https://drive.google.com/drive/folders/1DU9MaHUgckmV3tGLRp2Yz6u5i6iAf9U1.

· In the six scenes of the CBS dataset, there are FRF, NRF images, and reference images (GT) corresponding to FRF images. The FRF and NRF images contain five levels of distortion in each scenes. The FRF images are randomly sampled 150 times at each distortion level, the NRF images are randomly sampled 50 times at each distortion level. GT images can be used for rendering NVS scenes, the FRF and NRF images can be used for quality assessment of NVS scenes.

· FRF-MOS and NRF-MOS are the quality scores of images obtained from subjective experiments.

# MTSA model
· The MTSA model is now open source

· We use python = = 3.10, cuda = = 11.8 to build a virtual environment. 

· The libraries required by the virtual environment are listed in “requirements.txt”.

· If you want to run MTSA to train your dataset, you only need to run “MR.py”. However, it should be noted that you need to make a .txt file to represent the information of the image you want to input before training the dataset. The format can refer to the example given in the “QA _ list” folder.

· The pre-training model of feature extraction and NRF module can be obtained in https://drive.google.com/file/d/1n0JfkMzvLEIdgtV1j0Wcr0Upk8nwLu1q/view?usp=drive_link, https://drive.google.com/file/d/1hjTbXxFP4raC1HokCXKtH3oc5cUboXHA/view?usp=drive_link.
