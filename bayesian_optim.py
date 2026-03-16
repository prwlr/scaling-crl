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
  bo_iters: int = 3

  # Saving and loading
  should_save_state: bool = False
  should_load_state: bool = False
  save_path: str = DEFAULT_STATE_PATH
  load_path: str = DEFAULT_STATE_PATH

  # Non-optimised hyperparams
  num_epochs: int = 64
  num_envs: int = 128
  num_eval_envs: int = 128
  total_env_steps: int = 8 * BINARY_MILLION


def main():
  # Read args from commandline
  cli_args = tyro.cli(CliArgs)

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
        # Not to be optimised
        num_epochs=cli_args.num_epochs,
        num_envs=cli_args.num_envs,
        num_eval_envs=cli_args.num_eval_envs,
        total_env_steps=cli_args.total_env_steps,
        # To be optimised
        batch_size=batch_size,
        unroll_length=unroll_length,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
      )
    )

  param_bounds = {
    "x": (2, 4),
    "y": (-3, 3, int),
    "k": ("1", "2"),
  }

  optimizer = BayesianOptimization(
    f=train_optim_wrapper,
    pbounds=param_bounds,
    random_state=cli_args.seed,
    verbose=cli_args.logging_level
  )

  if cli_args.should_load_state:
    optimizer.load_state(cli_args.load_path)

  # Points to manually probe
  probes = [
    {"x": 0.5, "y": 2, "k": "1"},
    {"x": 0.5, "y": 2, "k": "1"},
  ]

  # Queues points to be probed upon maximize()
  for probe in probes:
    optimizer.probe(
      params=probe,
      lazy=True,
    )

  # Optim
  optimizer.maximize(
    init_points=cli_args.random_init_probes,
    n_iter=cli_args.bo_iters,
  )

  # The best combination of parameters and target value found
  print("Best target:", optimizer.max["target"])
  print("with params:", {key: round(value.item(), ndigits=3) for key, value in optimizer.max["params"].items()})

  # Save state
  if cli_args.should_save_state:
    optimizer.save_state(cli_args.save_path)

if __name__ == "__main__":
  main()
