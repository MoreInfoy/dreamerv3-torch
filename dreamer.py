import collections
import copy
import functools
import os
import pathlib
import sys
from envs.prometheus.core.env.humanoid import HumanoidTaskEnv
from envs.prometheus.core.sim.isaacsim import IsaacSim
from envs.prometheus.core.task.locomotion import Locomotion

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from envs.prometheus import SOURCE_DIR
import time

to_np = lambda x: x.detach().cpu().numpy()


def count_iters(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.pt"))


def make_env(config):
    sim = IsaacSim(config.sim)
    env = HumanoidTaskEnv(sim, Locomotion(sim, config.task), config.env)
    return env


class Dreamer(nn.Module):
    _wm: models.WorldModel
    _task_behavior: models.ImagBehavior

    def __init__(self, obs_space, act_space, config, logger):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_expl = tools.Until(config.expl_until)
        self._metrics = {}
        # this is update step
        self._update_count = 0
        self._wm = models.WorldModel(obs_space, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
                config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, state, training=False):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._update_count):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def train_model(self, data):
        metrics = {}
        for _ in range(self._config.num_learning_epochs):
            post, context, mets = self._wm.train_model(copy.deepcopy(data))
        metrics.update(mets)
        start = {k: v[:, 0].unsqueeze(1) for k, v in post.items()}
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        for _ in range(self._config.num_learning_epochs):
            metrics.update(self._task_behavior.train_model(copy.deepcopy(start), reward)[-1])

        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

        self._update_count += 1
        self._metrics["update_count"] = self._update_count
        if self._should_log(self._update_count):
            for name, values in self._metrics.items():
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []
            if self._config.video_pred_log:
                openl = self._wm.video_pred(data)
                self._logger.video("train_openl", to_np(openl))
            self._logger.write(iters=self._update_count)


@hydra.main(version_base=None, config_name="config", config_path=os.path.join(SOURCE_DIR, "cfg"))
def main(cfg: DictConfig) -> None:
    config = cfg.train
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # step in logger is environmental step
    logger = tools.Logger(logdir)

    print("Create envs.")
    train_eps = collections.OrderedDict()
    eval_eps = collections.OrderedDict()
    env = make_env(cfg)
    acts = env.action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])

    print("Start training.")
    training_agent = functools.partial(agent, training=True)

    # make sure eval will be executed once after config.steps
    for iter_ in range(config.max_iterations):
        start = time.time()

        state = tools.simulate(
            training_agent,
            env,
            train_eps,
            logger,
            episode_length=config.episode_length,
            state=state,
        )

        episodes = next(iter(train_eps.values()))
        data = {
            k: torch.stack(v, dim=1) for k, v in episodes.items() if "log" not in k
        }
        print(f"time cost for data collection: {time.time() - start} s")
        start = time.time()

        agent.train_model(data)
        print(f"time cost for model training: {time.time() - start} s")

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / f"model_{iter_}.pt")
        tools.clean_cache(train_eps, env.id)

    # save latest model
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
