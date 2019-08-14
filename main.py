from src.config import base_config
from src.dqn_agent import DQNAgent
from src.drqn_agent import DRQNAgent
from src.goru_agent import GORUAgent

class Main():
    def __init__(self, network_type, conf):
        if network_type=='dqn':
            print('DQN')
            self.agent=DQNAgent(conf)
        elif network_type=='drqn':
            print('LSTM-DRQN')
            self.agent=DRQNAgent(conf)
        elif network_type=='goru':
            print('GORU-DRQN')
            self.agent=GORUAgent(conf)
        else:
            raise ValueError('Incompatible network type '+network_type)

    def train(self, steps):
        self.agent.train(steps)

    def play(self, episodes, net_path):
        self.agent.play(episodes, net_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--network_type", type=str, default="dqn", help="Type of the network to build, can either be 'dqn' or 'drqn' or 'goru'")
    parser.add_argument("--train", type=str, default="True", help="Whether to train a network or to play with a given network")
    parser.add_argument("--model_dir", type=str, default="saved_session/net/", help="directory to save the model and replay memory during training")
    parser.add_argument("--net_path", type=str, default="", help="path to checkpoint of model")
    parser.add_argument("--steps", type=int, default=10000000, help="number of frames to train")
    args, remaining = parser.parse_known_args()

    conf=base_config()
    conf.network_type = args.network_type
    conf.train = args.train
    conf.dir_save = args.model_dir
    conf.train_steps = args.steps
    main = Main(conf.network_type, conf)

    if conf.train == "True":
        print(conf.train)
        main.train(conf.train_steps)
    else:
        assert args.net_path != "", "Please specify a net_path using the option --net_path"
        main.play(500, args.net_path)