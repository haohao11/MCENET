# Context Conditional Variational Autoencoder for Predicting Multi-Path Trajectories in Mixed Traffic

Trajectory prediction in urban mixed-traffic zones is critical for many AI systems, such as traffic management, social robots and autonomous driving. However, there are many challenges to predict the trajectories of heterogeneous road agents (pedestrians, cyclists and vehicles) at a microscopic-level. For example, an agent might be able to choose multiple plausible paths in complex interactions with other agents in varying environments. To this end, we propose an approach named **Context Conditional Variational Autoencoder (CCVAE)** that encodes both past and future scene context, interaction context and motion information to capture the variations of the future trajectories using a set of stochastic latent variables. We predict multi-path trajectories conditioned on past information of the target agent by sampling the latent variable multiple times. Through experiments on several datasets of varying scenes, our method outperforms the recent state-of-the-art methods for mixed traffic trajectory prediction by a large margin and more robust in a very challenging environment.

![CCVAE_pipeline](https://github.com/haohao11/CCVAE/blob/master/CCVAE_pipeline.png)


## Code usage
Install required packages, see requirements.txt

generate data
```python
python dparser.py
```

Train the model
There are two ways to train the model:
```python
python ccvae_mixed.py
```

## Download preprocessed data and pretrained model
Download preprocessed data

Download pretrained model

Run the pretrained model
There are two ways to train the model:
```python
python ccvae_mixed.py --train_mode False
```


## Citation
```
@misc{cheng2020context,
    title={Context Conditional Variational Autoencoder for Predicting Multi-Path Trajectories in Mixed Traffic},
    author={Hao Cheng and Wentong Liao. Michael Ying Yang and Monica Sester and Bodo Rosenhahn},
    year={2020},
    eprint={2002.05966},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

If you use HC dataset, please cite
```
@inproceedings{cheng2019pedestrian,
  title={Pedestrian Group Detection in Shared Space},
  author={Cheng, Hao and Li, Yao and Sester, Monika},
  booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)},
  pages={1707--1714},
  year={2019},
  organization={IEEE}
}
```


