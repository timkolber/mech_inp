from src.data_processing import data
from src.utils.logging import setup_logging


def main() -> None:
    logger = setup_logging()
    logger.debug("Logger set up successfully, starting main function.")
    go_dataset = data.GoDataset("cgos")
    print(go_dataset.sequences)


if __name__ == "__main__":
    main()
