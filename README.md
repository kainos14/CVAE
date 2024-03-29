# 


>**This is the official implementation of the paper with the title “Fall Detection of the Elderly Using Denoising LSTM-based Convolutional Variant Autoencoder”by Myung-Kyu Yi, KyungHyun Han, and Seong Oun Hwang**

## Paper Overview

**Abstract**: Globally, every country is seeing an increase in the number and percentage of older people. Due to this reason, there is a growing interest in health issues that could compromise the safety and quality of life for the elderly. Among these health concerns, falls stand out as a significant issue that significantly affects the elderly, and numerous studies have been conducted to address this problem. Nevertheless, gathering data on fall activities and integrating the fall detection solution into wearable devices remains challenging. The fall data collecting problem can be solved by the unsupervised learning method, which requires a complex deep learning model with many parameters, making integrating into wearable devices hard. Building an unsupervised deep learning model with fewer parameters while keeping the equivalent high accuracy is desirable. In this paper, we propose a novel fall detection method using unsupervised learning based on the denoising Long Short Term Memory (LSTM)-based Convolutional Variant Autoencoder (CVAE) model with fewer parameters, maintaining the equivalent accuracy by the proposed data debugging and hierarchical data balancing techniques. Experimental results show that our method can achieve an F1-score of 1.0 while reducing parameters by 25.6 times (resulting in a memory size of 157.65 KB suitable for wearable devices) compared to the state-of-the-art unsupervised deep learning method.

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

@inJournal{XXX,
  title={Fall Detection of the Elderly Using Denoising LSTM-based Convolutional Variant Autoencoder},
  author={Myung-Kyu Yi, KyungHyun Han, and Seong Oun Hwang},
  booktitle={IEEE XXX},
  year={2023}
}

```

## Contact

Please feel free to contact via email (<kainos@gachon.ac.kr>) if you have further questions.
