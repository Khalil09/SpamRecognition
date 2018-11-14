import argparse
from adapter import Adapter

DEFAULT_ALGORITHM = 'mlp'

def getAdapter():

    parser = argparse.ArgumentParser(description='An Artificial Neural Network model comparison using MLP and FBR to recognise spam.')

    parser.add_argument('--algorithm', dest='algorithm', choices=['mlp', 'fbr'], type=str, default=DEFAULT_ALGORITHM, help='Select Learning Algorithm')
    #
    # parser.add_argument('-cv', dest='crossVal', type=int, default=DEFAULT_CROSSVAL, help='Select Cross Validation Value')
    #
    # parser.add_argument('-n', dest='number_of_trees', type=int, default=DEFAULT_NUMBER_OF_TREES, help='Number of trees in forest')
    #
    # parser.add_argument('--train', dest='train', default=DEFAULT_TRAIN, type=bool, help='Enable or disable the training of the model')
    #
    # parser.add_argument('--plot', dest='confPlot', default=DEFAULT_PLOT ,type=bool, help='Change how confusion matrices are plot.')
    #
    # parser.add_argument('-b', dest='best', default=DEFAULT_BEST ,type=bool, help='Use only the 5 best features')
    #
    args = parser.parse_args()
    #
    adapter = Adapter(args.algorithm)

    return adapter
