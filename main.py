from absl import app, flags
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch
from nn import get_nn
from data import get_data
from optimizers import DPGroupFTRLProximalOptimizer
import utils
from utils import EasyDict


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], 'Dataset selection')
flags.DEFINE_boolean('dp_ftrl', True, 'Use DP-GFTPRL or vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 4.0, 'Noise ratio.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')
flags.DEFINE_integer('restart', 0, 'Restart tree every X epochs.')
flags.DEFINE_boolean('effi_noise', False, 'Use efficient tree aggregation.')
flags.DEFINE_boolean('tree_completion', False, 'Generate noise until power of 2.')
flags.DEFINE_float('momentum', 0, 'Momentum.')
flags.DEFINE_float('learning_rate', 0.4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 250, 'Batch size.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs.')
flags.DEFINE_integer('run', 1, 'Random seed run.')
flags.DEFINE_string('dir', '.', 'Results directory.')

def main(argv):
    # Set random seed and configure GPU
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda')

    # Load dataset
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    batch_size = FLAGS.batch_size or ntrain
    num_batches = ntrain // batch_size
    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg

    log_dir = os.path.join(FLAGS.dir, FLAGS.data, utils.get_fn(
        EasyDict(batch=batch_size),
        EasyDict(dpsgd=FLAGS.dp_ftrl, restart=FLAGS.restart, completion=FLAGS.tree_completion),
        EasyDict(lr=FLAGS.learning_rate, momentum=FLAGS.momentum),
        EasyDict(run=FLAGS.run)
    ))

    print(f'Model dir: {log_dir}')

    # Prepare data streaming
    class DataStream:
        def __init__(self):
            self.perm = np.random.permutation(ntrain)
            self.i = 0

        def __call__(self):
            batch_idx = self.perm[self.i * batch_size:(self.i + 1) * batch_size]
            self.i = (self.i + 1) % num_batches
            return trainset.image[batch_idx], trainset.label[batch_idx]

    data_stream = DataStream()

    # Set optimizer and noise generator
    model = get_nn(FLAGS.data, nclass).to('cuda')
    optimizer = DPGroupFTRLProximalOptimizer(model.parameters(), lr=FLAGS.learning_rate,
                              momentum=FLAGS.momentum, noise_multiplier=FLAGS.noise_multiplier,
                              l2_norm_clip=FLAGS.l2_norm_clip)

    def get_cumm_noise():
        if not FLAGS.dp_ftrl or FLAGS.noise_multiplier == 0:
            return lambda: [torch.zeros((1,)).to('cuda')]
        return (CummuNoiseEffTorch if FLAGS.effi_noise else CummuNoiseTorch)(
            FLAGS.noise_multiplier * FLAGS.l2_norm_clip / batch_size,
            [p.size() for p in model.parameters()],
            'cuda'
        )

    cumm_noise = get_cumm_noise()

    # Training loop
    def train_loop(epoch):
        model.train()
        losses = []
        criterion = torch.nn.CrossEntropyLoss()
        for _ in trange(num_batches, desc=f'Epoch {epoch+1}/{FLAGS.epochs}', leave=False):
            data, target = map(torch.Tensor, data_stream())
            data, target = data.to('cuda'), target.long().to('cuda')

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step((FLAGS.learning_rate, cumm_noise()))
            losses.append(loss.item())

        print(f'Epoch {epoch+1} Loss: {np.mean(losses):.2f}')

    # Test loop
    def test():
        model.eval()
        accs = []
        with torch.no_grad():
            for dataset in [trainset, testset]:
                correct = 0
                for i in range(0, dataset.image.shape[0], 1000):
                    data, target = dataset.image[i:i+1000], dataset.label[i:i+1000]
                    data, target = torch.Tensor(data).to('cuda'), torch.LongTensor(target).to('cuda')
                    pred = model(data).argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                accs.append(correct / dataset.image.shape[0])
        return accs

    writer = SummaryWriter(log_dir)
    for epoch in range(FLAGS.epochs):
        train_loop(epoch)
        train_acc, test_acc = test()
        writer.add_scalar('eval/train_accuracy', 100 * train_acc, epoch)
        writer.add_scalar('eval/test_accuracy', 100 * test_acc, epoch)
        if FLAGS.restart and (epoch + 1) % FLAGS.restart == 0:
            optimizer.restart()
            cumm_noise = get_cumm_noise()
    writer.close()

if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
