import pandas as pd
import itertools


modes = ["opt", "full_ft", "scmr"]
target_gatesets = ["CLIFFORDT", "IBMN", "NAM", "IBMO", "ION"]
optimization_objectives = ["FT", "FIDELITY", "TWO_Q", "TOTAL"]
opt_timeouts = [60]
approx_epsilons = [0, 1e-8]
architectures = ["square_sparse_layout", "compact_layout"]
advanced_args = [None, "test_advanced_args_1.json"]

# These are not the values to use as args. Rather, if True then pass the flag, else don't.
verbose = [True, False]
guoq_help = [True, False]

result = []

for (
    mode,
    target_gateset,
    optimization_objective,
    opt_timeout,
    approx_epsilon,
    architecture,
    advanced_args,
    verbose,
    guoq_help,
) in itertools.product(
    modes,
    target_gatesets,
    optimization_objectives,
    opt_timeouts,
    approx_epsilons,
    architectures,
    advanced_args,
    verbose,
    guoq_help,
):
    result.append(
        {
            "mode": mode,
            "target_gateset": target_gateset,
            "optimization_objective": optimization_objective,
            "opt_timeout": opt_timeout,
            "approx_epsilon": approx_epsilon,
            "architecture": architecture,
            "advanced_args": advanced_args,
            "verbose": verbose,
            "guoq_help": guoq_help,
        }
    )

df = pd.DataFrame(result)
df.to_csv("args.csv", index=False)

guoq_help_false = df[df["guoq_help"] == False]
guoq_help_false.sample(10).to_csv("args_limited.csv", index=False)
