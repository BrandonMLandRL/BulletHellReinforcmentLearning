# This file will handle creating a TCP server that Actors will connect to which the Actors will send experience tuples to.
# Learner will then train using DQN on the incoming data. Learner will broadcast to all Actors when it has finished updating the weights file.
# Actors should use a counting semaphore to prevent the Learner from updating the weights again while an Actor is reading still.

from bullet_hell_rl.DQN.LearnerServerComponent import (
    HOST,
    PORT,
    LearnerServerComponent,
)


def main(host: str = HOST, port: int = PORT) -> None:
    LearnerServerComponent(host=host, port=port).start_server()


if __name__ == "__main__":
    main()
