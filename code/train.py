from args import parse_train_opt
from badm import badm


def train(opt):
    model = badm(opt.feature_type)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
