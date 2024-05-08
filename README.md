# 


>**This is the official implementation of the paper with the title “Fall Detection of the Elderly Using Denoising LSTM-based Convolutional Variant Autoencoder”by Myung-Kyu Yi, KyungHyun Han, and Seong Oun Hwang**
>
>https://ieeexplore.ieee.org/document/10508748
>
>https://doi.org/10.1109/JSEN.2024.3388478

## Paper Overview

**Abstract**: 
As societies age, the issue of falls has become increasingly critical for the health and safety of the elderly. Fall detection in the elderly has traditionally relied on supervised learning methods, which require data on falls, which is difficult to obtain in real situations. Additionally, the complexity of integrating deep learning models into wearable devices for real-time fall detection has been challenging due to limited computational resources. In this paper, we propose a novel fall detection method using unsupervised learning based on a denoising long short term memory (LSTM)-based convolutional variational autoencoder (CVAE) model to solve the problem of lack of fall data. By utilizing the proposed data debugging and hierarchical data balancing techniques, the proposed method achieves an F1 score of 1.0 while reducing the parameter count by 25.6 times compared to the state-of-the-art unsupervised deep learning method. The resulting model occupies only 157.65 KB of memory, making it highly suitable for integration into wearable devices.

---
## Dataset
- MobiFall dataset is available at https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
- MobiAct dataset is available at https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
- SisFall dataset is available at http://sistemic.udea.edu.co/investigacion/proyectos/english-falls/?lang=en
- FallAllD dataset is available at http://10.21227/bnya-mn34

## Codebase Overview
- We note that:
  - <CVAE.py> for the proposed DCVAE model.

Framework uses Tensorflow 2+, tensorflow_addons, numpy, pandas, matplotlib, scikit-learn.  
  
## Citing This Repository

If our project is helpful for your research, please consider citing :

```
@article{yi2024jsen,
  title={Fall Detection of the Elderly Using Denoising LSTM-based Convolutional Variant Autoencoder},
  author={Myung-Kyu Yi, KyungHyun Han, and Seong Oun Hwang},
  journal={IEEE Sensor Journal},
  volume={},
  number={},
  pages={},
  year={2024}
  publisher={IEEE}
}

```

## Contact

Please feel free to contact via email (<kainos@gachon.ac.kr>) if you have further questions.
