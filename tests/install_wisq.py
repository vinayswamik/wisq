import subprocess
import sys
from packaging import version


def install_build():
    """Checks if the 'build' module is installed and installs it if not."""
    try:
        import build
    except ImportError:
        print("Installing 'build' module...")
        subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)


def get_python_version():
    return sys.version.split()[0]


def check_version(installed, required, name):
    if version.parse(installed) < version.parse(required):
        print(
            f"Error: {name} {required} or higher is required. Your version is {installed}."
        )
        sys.exit(1)


def run_command(command, error_message):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(error_message)
        sys.exit(1)


def main():
    REQUIRED_PYTHON = "3.8"

    install_build()

    python_version = get_python_version()

    check_version(python_version, REQUIRED_PYTHON, "Python")

    print("Building and installing wisq...")

    run_command([sys.executable, "-m", "build", "--sdist"], "Build failed.")
    run_command(["pip3", "uninstall", "wisq", "-y"], "Failed to uninstall wisq.")
    run_command(["pip3", "install", "dist/wisq-0.0.1.tar.gz"], "Installation failed.")

    print("Installation completed successfully.")


if __name__ == "__main__":
    main()
