import os
import sys
import json
import pandas as pd
import subprocess
import time


def build_flags(row):
    flags = [
        "--mode",
        row["mode"],
        "--target_gateset",
        row["target_gateset"],
        "--optimization_objective",
        row["optimization_objective"],
        "--opt_timeout",
        str(row["opt_timeout"]),
        "--approx_epsilon",
        str(row["approx_epsilon"]),
        "--architecture",
        row["architecture"],
    ]

    if pd.notna(row["advanced_args"]):
        flags.extend(["--advanced_args", row["advanced_args"]])

    if row.get("verbose", False):
        flags.append("--verbose")
    if row.get("guoq_help", False):
        flags.append("--guoq_help")

    return flags


def run_wisq_benchmark(circuit_file, flags):
    """Runs the wisq benchmark on a single circuit file with specified flags."""
    if not os.path.isfile(circuit_file):
        print(f"Error: Circuit file '{circuit_file}' not found.")
        return {
            "circuit_file": circuit_file,
            "flags": " ".join(flags),
            "success": False,
            "time": None,
            "stdout": "",
            "stderr": "",
        }

    print(f"Running wisq benchmark on '{circuit_file}'...")
    print(f"Using wisq options: {flags}")

    start_time = time.time()
    result = subprocess.run(
        ["wisq", circuit_file] + flags, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    duration = time.time() - start_time

    mode = flags[flags.index("--mode") + 1]
    success = False

    if mode == "opt" and os.path.exists("out.qasm"):
        success = True
        os.remove("out.qasm")
    elif mode in {"scmr", "full_ft"} and os.path.exists("out.json"):
        success = True
        os.remove("out.json")

    return {
        "circuit_file": circuit_file,
        "flags": " ".join(flags),
        "success": success,
        "time": duration,
        "stdout": result.stdout.decode("utf-8").strip(),
        "stderr": result.stderr.decode("utf-8").strip(),
    }


def run_all(folder, args_file, output_file="test_results.json"):
    """Processes all circuits with all flag combinations and saves results to a JSON file."""
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' not found.")
        sys.exit(1)

    if not os.path.isfile(args_file):
        print(f"Error: Args file '{args_file}' not found.")
        sys.exit(1)

    circuits = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".qasm")
    ]
    df = pd.read_csv(args_file)
    results = []

    for circuit in circuits:
        for _, row in df.iterrows():
            flags = build_flags(row)
            result = run_wisq_benchmark(circuit, flags)
            results.append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"All benchmarks completed. Results saved to {output_file}.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_all.py <folder_of_circuits> <args_file>")
        sys.exit(1)

    folder = sys.argv[1]
    args_file = sys.argv[2]
    run_all(folder, args_file)


if __name__ == "__main__":
    main()
