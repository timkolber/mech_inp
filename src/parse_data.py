from logging import DEBUG

from data_processing import data
from utils.logging import setup_logging


def main() -> None:
    logger = setup_logging(log_filename="main.log", log_level=DEBUG)
    logger.debug("Logger set up successfully, starting main function.")
    train_go_dataset = data.GoDataset("cgos", split="train")
    val_go_dataset = data.GoDataset("cgos", split="val")
    print(len(train_go_dataset.games))
    print(len(val_go_dataset.games))


if __name__ == "__main__":
    main()
