from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np

# from train import main


def black_box_function(x, y, k):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + (int(k) * 4)


if __name__ == "__main__":
    # Parameter bounds,
    pbounds = {
        "x": (2, 4),
        "y": (-3, 3, int), # Supply type for contrained optim
        "k": ("1", "2"), # Categorical params via strings
    }

    # Create the optimiser
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # Do optim
    optimizer.maximize(
        init_points=2, # Num random points to probe before optim
        n_iter=3, # Num bayesian optim steps
    )

    # The best combination of parameters and target value found
    initial_best = optimizer.max
    print("Initial best target:", initial_best["target"])
    print("with params:", {key: round(value.item(),ndigits=3) for key, value in initial_best["params"].items()})

    # List of all parameters probed and their corresponding target values
    # for i, res in enumerate(optimizer.res):
    #     print(f"Iteration {i}: \n\t{res}")

    # Setting new bounds
    # optimizer.set_bounds(new_bounds={"x": (-2, 3)})

    # Manually probe point
    # optimizer.probe(
    #     params={"x": 0.5, "y": 2, "k": "1"},
    #     lazy=True,
    # )

    # Re-run optim without random probing
    optimizer.maximize(
        init_points=0,
        n_iter=5,
    )

    # The best combination of parameters and target value found
    new_best = optimizer.max
    print("New best target:", new_best["target"])
    print("with params:", {key: round(value.item(), ndigits=3) for key, value in new_best["params"].items()})

    # Get the probed points for scatter
    res = optimizer._space.res()
    k1 = np.array([[p['params']['x'], p['params']['y']] for p in res if p['params']['k']=='1'])
    k2 = np.array([[p['params']['x'], p['params']['y']] for p in res if p['params']['k']=='2'])

    # Plot the actual function
    x1 = np.linspace(pbounds['x'][0], pbounds['x'][1], 1000)
    x2 = np.linspace(pbounds['y'][0], pbounds['y'][1], 1000)

    X1, X2 = np.meshgrid(x1, x2)
    Z1 = black_box_function(X1, X2, '1')
    Z2 = black_box_function(X1, X2, '2')

    fig, axs = plt.subplots(1, 2)

    vmin = np.min([np.min(Z1), np.min(Z2)])
    vmax = np.max([np.max(Z1), np.max(Z2)])

    axs[0].contourf(X1, X2, Z1, vmin=vmin, vmax=vmax)
    axs[0].set_aspect("equal")
    axs[0].scatter(k1[:,0], k1[:,1], c='k')
    axs[1].contourf(X1, X2, Z2, vmin=vmin, vmax=vmax)
    axs[1].scatter(k2[:,0], k2[:,1], c='k')
    axs[1].set_aspect("equal")
    axs[0].set_title('k=1')
    axs[1].set_title('k=2')
    fig.tight_layout()

    # Show the function(s)
    plt.show()

    # # Save state
    # optimizer.save_state("optimizer_state.json")
    #
    # # Re-load state
    # new_optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds={"x": (-2, 3), "y": (-3, 3)},
    #     random_state=1,
    #     verbose=0
    # )
    # new_optimizer.load_state("./optimizer_state.json")
    #
    # # Continue optimization
    # new_optimizer.maximize(
    #     init_points=0,
    #     n_iter=5
    # )


