from bayes_opt import BayesianOptimization
from dataclasses import dataclass, asdict
import tyro
import train

DEFAULT_STATE_PATH = "./bayes_optimizer_state.json"
BINARY_MILLION = 2**20


@dataclass
class CliArgs:
  # General
  seed: int = 1000
  logging_level: int = 2
  random_init_probes: int = 0
  bo_iters: int = 16

  # Saving and loading
  should_save_state: bool = True
  should_load_state: bool = True
  save_path: str = DEFAULT_STATE_PATH
  load_path: str = DEFAULT_STATE_PATH

  # Non-optimised hyperparams
  num_epochs: int = 64
  num_envs: int = 128
  num_eval_envs: int = 128
  total_env_steps: int = 16 * BINARY_MILLION

  # Target metric to optimise for
  target_metric: str = "eval/episode_reward"
  target_denominator: str = "training/walltime"


def main():
  # Read args from commandline
  cli_args = tyro.cli(CliArgs)

  # Parameter bounds and typing/contraints
  param_bounds = {
    "batch_size": (16, 2048, int),
    "unroll_length": (8, 256, int),
    "actor_lr": (3e-6, 3e-3),
    "critic_lr": (3e-6, 3e-3),
    "alpha_lr": (3e-6, 3e-3),
  }

  # Points to manually probe
  probes = [
    {"batch_size": 128, "unroll_length": 64, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4},
    {"batch_size": 256, "unroll_length": 128, "actor_lr": 3e-5, "critic_lr": 3e-5, "alpha_lr": 3e-5}
    # Loosly based on best in previous optim round
    {"batch_size": 2048, "unroll_length": 256, "actor_lr": 3e-3, "critic_lr": 3e-5, "alpha_lr": 3e-5}
  ]

  # Wrap train.main with a function that only takes the params to be optimised
  def train_optim_wrapper(
    batch_size: int,
    unroll_length: int,
    actor_lr: float,
    critic_lr: float,
    alpha_lr: float,
  ) -> float:

    return train.main(
      train.Args(
        # Always same
        capture_vis=False,
        checkpoint=False,
        # Not to be optimised
        num_epochs=cli_args.num_epochs,
        num_envs=cli_args.num_envs,
        num_eval_envs=cli_args.num_eval_envs,
        total_env_steps=cli_args.total_env_steps,
        return_metric=cli_args.target_metric,
        return_denominator=cli_args.target_denominator,
        # To be optimised
        batch_size=batch_size,
        unroll_length=unroll_length,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
      )
    )

  # Create the optimiser
  optimizer = BayesianOptimization(
    f=train_optim_wrapper,
    pbounds=param_bounds,
    random_state=cli_args.seed,
    verbose=cli_args.logging_level
  )

  if cli_args.should_load_state:
    optimizer.load_state(cli_args.load_path)

  # Queues points to be probed upon maximize()
  for probe in probes:
    optimizer.probe(
      params=probe,
      lazy=True,
    )

  # Do optimisation
  optimizer.maximize(
    init_points=cli_args.random_init_probes,
    n_iter=cli_args.bo_iters,
  )

  # The best target value and combination of parameters that was found
  print("Best target:", optimizer.max["target"])
  print("with params:", {key: round(value.item(), ndigits=3) for key, value in optimizer.max["params"].items()})

  # Save state
  if cli_args.should_save_state:
    optimizer.save_state(cli_args.save_path)

if __name__ == "__main__":
  main()
