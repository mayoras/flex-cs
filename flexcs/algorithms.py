from typing import Tuple

from flex.pool import FlexPool

from .server import (
    copy_server_model_to_clients,
    get_clients_weights,
    aggregate_uniform_fedavg,
    set_agreggated_weights_to_server,
    evaluate_global_model,
)
from .client import train


def random_sampling_run(
    pool: FlexPool, rounds: int = 20, clients_per_round: int = 20
) -> Tuple[list, list]:

    losses = []
    accuracies = []

    for t in range(rounds):
        print(f"\nRunning round: {t+1} of {rounds}")

        # select clients randomly
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients

        print(f"Selected clients for this round: {len(selected_clients)}")

        # Deploy the server model to the selected clients
        pool.servers.map(copy_server_model_to_clients, selected_clients)

        # Each selected client trains her model
        selected_clients.map(train)

        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(aggregate_uniform_fedavg)

        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)
        metrics = pool.servers.map(evaluate_global_model)

        # append metrics
        loss, acc = metrics[0]
        losses.append(loss)
        accuracies.append(acc)
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")

        return losses, accuracies
