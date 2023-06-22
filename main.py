r"""
"""

from src.utils import get_logger, seed_everything
from src.dataset import FedAvgRetriever
from src.modules import FedAvgClient, FedAvgServer

config = {
    "batch_size": 128,
    "device": "cuda:0",
    "epochs": 5,
    "learning_rate": 0.0001,
    "num_clients": 5,
    "rounds": 5,
    "seed": 42,
}


def main():

    seed_everything(config["seed"])
    logger = get_logger()

    logger.info("FedAvg start")

    # init dataset
    retriever = FedAvgRetriever(config)
    dataloaders = retriever.get(config["num_clients"])

    # init server and clients
    server = FedAvgServer(config)
    clients = [FedAvgClient(config, dataloaders[i])
               for i in range(config["num_clients"])]

    # init FedAvg
    global_model = server.init_global_model()
    for client in clients:
        client.set_global_model(global_model)

    # round loop
    for round in range(config["rounds"]):

        logger.info(f"Round: {round} start")

        local_models = [client.train_local_model() for client in clients]

        global_model = server.aggregate(local_models)
        for client in clients:
            client.set_global_model(global_model)

        logger.info(f"Round: {round} finished")

    logger.info("FedAvg finished")


if __name__ == "__main__":
    main()
