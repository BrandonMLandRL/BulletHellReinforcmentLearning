# This file will handle creating a TCP server that Actors will connect to which the Actors will send experience tuples to.
# Learner will then train using DQN on the incoming data. Learner will broadcast to all Actors when it has finished updating the weights file.
# Actors should use a counting semaphore to prevent the Learner from updating the weights again while an Actor is reading still.

from collections import deque 
import queue 

from bullet_hell_rl.DQN.LearnerServerComponent import (
    HOST,
    PORT,
    LearnerServerComponent,
)
from ..net.protocol import (
    MSG_EXPERIENCE_TUPLE,
    MSG_WEIGHTS_READY,
    MSG_WEIGHTS_READY_ACK
)

class Learner:
    def __init__(self,lsc:LearnerServerComponent):
        #Create the network and everything training related to the Legacy DQN here.
        self.replayBufferSize = 20000
        #Step 1: Get the replay buffer to fill with data from the Actor. 
        self.replay_buffer = deque(maxlen=self.replayBufferSize)
        pass
    
    def _on_message_recv_thread(self,msg):
        while not self.stop_event.is_set():
            try:
                msg = self.lsc._recv_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                if not isinstance(msg, dict):
                    continue

                mtype = msg.get("type")
                if mtype == MSG_EXPERIENCE_TUPLE:
                    state = msg["state"]
                    action = int(msg["action"])
                    reward = float(msg["reward"])
                    next_state = msg["next_state"]
                    done = bool(msg["done"])
                    meta = msg.get("meta", {})

                elif mtype == MSG_WEIGHTS_READY_ACK:
                    #TODO: Decrement the count of clients we are waiting for.
                    pass

                else:
                    print(f"Unexpected Learner msg type: {mtype}")

            except (KeyError, TypeError, ValueError) as e:
                print(f"Bad Message {e}")
       

def main(host: str = HOST, port: int = PORT) -> None:
    LearnerServerComponent(host=host, port=port).start_server()


if __name__ == "__main__":
    main()
