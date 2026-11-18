import sys
from src.gui import launch_gui
from src.experiments import run_benchmark
from src.evolution import GeneticTrainer


def main():
    if "--benchmark" in sys.argv:
        run_benchmark()
    elif "--train" in sys.argv:
        GeneticTrainer().train()
    else:
        launch_gui()


if __name__ == "__main__":
    main()