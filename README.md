# CCVAE

Trajectory prediction in urban mixed-traffic zones is critical for many AI systems, such as traffic management, social robots and autonomous driving. However, there are many challenges to predict the trajectories of heterogeneous road agents (pedestrians, cyclists and vehicles) at a microscopic-level. For example, an agent might be able to choose multiple plausible paths in complex interactions with other agents in varying environments. To this end, we propose an approach named \emph{Context Conditional Variational Autoencoder} (CCVAE) that encodes both past and future scene context, interaction context and motion information to capture the variations of the future trajectories using a set of stochastic latent variables. We predict multi-path trajectories conditioned on past information of the target agent by sampling the latent variable multiple times. Through experiments on several datasets of varying scenes, our method outperforms the recent state-of-the-art methods for mixed traffic trajectory prediction by a large margin and more robust in a very challenging environment.
