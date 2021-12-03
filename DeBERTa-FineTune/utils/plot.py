import os
import matplotlib.pyplot as plt


def plot_line(x_train, y_train, x_val=None, y_val=None, item='Loss', out_dir=None, name='plot'):
    """
    plot loss or metric curve line, output to the specified directory.
    Args:
        :param x_train: iterable, range of training steps
        :param y_train: iterable, traning item
        :param x_val:   iterable, range of validation steps
        :param y_val:   iterable, validation item
        :param item:    str, indicate the item you plot, e.g 'Loss', 'Acc'
        :param out_dir: str, output directory
        :param name:    str, output file name
    Returns:
        out_path:    str, path to output file
    """

    assert out_dir is not None, "output directory must be specified!"

    plt.plot(x_train, y_train, label='Train')
    if x_val is not None and y_val is not None:
        plt.plot(x_val, y_val, label='Val')
    
    plt.xlabel('Step')
    plt.ylabel(item)

    # loc = 'upper right' if item == 'Loss' else 'upper left'
    # plt.legend(loc=loc)
    plt.legend(loc='best')
    plt.title('_'.join([item]))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.png')

    plt.savefig(out_path)
    plt.close()

    return out_path
