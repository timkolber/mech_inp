from logging import DEBUG

from data_processing import data
from utils.logging import setup_logging


def main() -> None:
    logger = setup_logging(log_filename="main.log", log_level=DEBUG)
    logger.debug("Logger set up successfully, starting main function.")
    go_dataset = data.GoDataset("cgos")
    print(go_dataset.sequences)


if __name__ == "__main__":
    main()
