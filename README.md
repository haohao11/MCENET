# MCENET: Multi-Context Encoder Network for Trajectory Prediction in Mixed Traffic

Trajectory prediction in urban mixed-traffic zones (a.k.a. shared spaces) is critical for many intelligent transportation systems, such as intent detection for autonomous driving. However, there are many challenges to predict the trajectories of heterogeneous road agents (pedestrians, cyclists and vehicles) at a microscopical level. For example, an agent might be able to choose multiple plausible paths in complex interactions with other agents in varying environments. To this end, we propose an approach named **Multi-Context Encoder Network** (MCENET) that is trained by encoding both past and future scene context, interaction context and motion information to capture the patterns and variations of the future trajectories using a set of stochastic latent variables. In inference time, we combine the past context and motion information of the target agent with samplings of the latent variables to predict multiple realistic trajectories in the feature. Through experiments on several datasets of varying scenes, our method outperforms some of the recent state-of-the-art methods for mixed traffic trajectory prediction by a large margin and more robust in a very challenging environment. The impact of each context is justified via ablation studies.

![MCENET](https://github.com/haohao11/MCENET/blob/master/mcenet.jpg)
Predicting the future trajectory (d) by observing the past trajectories (c) considering the scene (a) and grouping context (b). Three kinds of scene context: (1) aerial photograph provides overview of the environment, (2) segmented map defines the accessible areas respective to road agents' transport mode and (3) the motion heat map describes the prior of how different agents move. Different colors in (b)(c)(d) denote different agents or agent groups.

## Code usage
Install required packages, see [requirements.txt](https://github.com/haohao11/CCVAE/blob/master/requirements.txt)


Set the *data_process* as *True* and Train the model
```python
python trainer.py
```

## Citation
```
@misc{cheng2020context,
    title={MCENET: Multi-Context Encoder Network for Homogeneous Agent Trajectory Prediction in Mixed Traffic},
    author={Hao Cheng and Wentong Liao and Michael Ying Yang and Monika Sester and Bodo Rosenhahn},
    year={2020},
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

[MIT license](https://github.com/haohao11/CCVAE/blob/master/license.md)

