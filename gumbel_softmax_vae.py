# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import collections
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='S',
                    help='learning_rate (default: 1e-3)')
parser.add_argument('--margin', type=float, default=0.0, metavar='S',
                    help='margin')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')
parser.add_argument('--argmax_solver', type=str, default='identity',
                    help='Armax Differentiatin')
parser.add_argument('--latent_dim', type=int, default=30, metavar='N',
                    help='Late')
parser.add_argument('--categorical_dim', type=int, default=10, metavar='N',
                    help='Categorical dims along each latent dim.')
parser.add_argument('--importance_weights', type=int, default=20, metavar='N',
                    help='Importance weights.')
parser.add_argument('--experiment_directory', type=str, default='data/temp', metavar='N',
                    help='Directory to save the results.')

TEMP_MIN = 0.5
ANNEAL_RATE = 0.00003
EPS = 1e-20
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def sample_gumbel(shape):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return - torch.log(- torch.log(U + EPS) + EPS)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, indices = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, indices.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # samples = torch.argmax(logits + sample_gumbel(logits.size()), dim=-1)
    # y_hard = 1.0 * torch.nn.functional.one_hot(samples, num_classes=categorical_dim)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def _resolve_vector_batch(a, b):
    """Resolves vector a along b."""
    assert a.ndim == b.ndim == 3
    norm_b = torch.norm(b, dim=-1)[:, :, None]
    unit_vector_b = b / norm_b
    batch_projection = torch.sum(a * unit_vector_b, dim=-1)[:, :, None]
    parallel_component = batch_projection * unit_vector_b
    perpendicular_component = a - parallel_component
    return parallel_component, perpendicular_component


class Bbbprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        logits = logits + sample_gumbel(logits.size())
        samples = torch.argmax(logits, dim=-1)
        samples = 1.0 * torch.nn.functional.one_hot(
            samples, num_classes=args.categorical_dim)
        ctx.save_for_backward(logits, samples)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        lamb = 20
        logits, output = ctx.saved_tensors
        output_prime = torch.argmax(logits - lamb * grad_output, dim=-1)
        output_prime = 1.0 * torch.nn.functional.one_hot(
            output_prime, num_classes=args.categorical_dim)
        return (output - output_prime) / lamb


def process_grad(grads):
    _, perpendicular = _resolve_vector_batch(grads, torch.ones_like(grads))
    return perpendicular


class IdentityRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        margin = args.margin * (
            2 * (torch.rand(logits.size()) > 0.5) - 1)
        logits = margin + logits + sample_gumbel(logits.size())
        ctx.save_for_backward(logits)
        samples = torch.argmax(logits, dim=-1)
        samples = 1.0 * torch.nn.functional.one_hot(
            samples, num_classes=args.categorical_dim)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        # grad_output = process_grad(grad_output)
        _, perpendicular = _resolve_vector_batch(grad_output, logits)
        perpendicular = perpendicular * torch.norm(grad_output) / (
            1e-6 + torch.norm(perpendicular))
        return perpendicular


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # margin = args.margin * (2 * (torch.rand(logits.shape()) > 0.5) - 1)
        margin = 0
        samples = torch.argmax(margin + logits + sample_gumbel(logits.size()), dim=-1)
        samples = 1.0 * torch.nn.functional.one_hot(samples, num_classes=args.categorical_dim)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = process_grad(grad_output)
        return grad_output


class VAE_gumbel(nn.Module):
    def __init__(self, temp):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.latent_dim * args.categorical_dim)

        self.fc4 = nn.Linear(args.latent_dim * args.categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        print('Argmax Solver:', args.argmax_solver)
        if args.argmax_solver == 'identity':
            self.argmax_solver = Identity()
        elif args.argmax_solver == 'identity_rotation':
            self.argmax_solver = IdentityRotation()
        elif args.argmax_solver == 'bbbprop':
            self.argmax_solver = Bbbprop()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, temp, hard):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), args.latent_dim, args.categorical_dim)
        probabilities = F.softmax(q_y, dim=-1)
        if args.argmax_solver == 'gumbel':
            z = gumbel_softmax(torch.log(probabilities + EPS), temp, hard)
        else:
            z = self.argmax_solver.apply(torch.log(probabilities + EPS))
        return (
            self.decode(z.view(-1, args.latent_dim * args.categorical_dim)),
            probabilities.reshape(* q.size()))


def log_mean_exp(x, dim):
    """Compute the log(mean(exp(x), dim)) in a numerically stable manner.

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Returns:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """Compute the log(sum(exp(x), dim)) in a numerically stable manner.

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Returns:
        tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def duplicate(x, rep):
    """Duplicates x along dim=0.

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def record_negative_iwae_metrics(model, x, iw, metrics):
    """Computes the Importance Weighted Autoencoder Bound."""
    q = model.encode(x.view(-1, 784))
    q_y = q.view(q.size(0), args.latent_dim, args.categorical_dim)
    z_probabilities = F.softmax(q_y, dim=-1)
    z_probabilities = duplicate(z_probabilities, iw)
    z_logits = torch.log(z_probabilities + EPS)
    z = 1.0 * torch.nn.functional.one_hot(
        torch.argmax(z_logits + sample_gumbel(z_logits.size()), dim=-1),
        num_classes=args.categorical_dim)

    bernoulli_logits = model.decode(z.view(-1, args.latent_dim * args.categorical_dim))

    log_p_x = - cross_entropy_loss(bernoulli_logits, duplicate(x, iw), reduce_batch=False)
    log_p_z_given_x = (z * torch.log(z_probabilities + EPS)).sum([1, 2])
    log_p_z_prior = (z * torch.log(torch.ones_like(z) / args.categorical_dim)).sum([1, 2])
    elbo = (log_p_x + log_p_z_given_x - log_p_z_prior).reshape(iw, -1)
    metrics['- log_p_x'].append(
        - log_mean_exp(log_p_x.reshape(iw, -1),
                       dim=0).mean().detach().numpy())
    metrics['- log_p_z_given_x'].append(
        - log_mean_exp(log_p_z_given_x.reshape(iw, -1),
                       dim=0).mean().detach().numpy())
    metrics['log_p_z_prior'].append(
        log_mean_exp(log_p_z_prior.reshape(iw, -1),
                     dim=0).mean().detach().numpy())
    metrics['niwae'].append(
        - log_mean_exp(elbo, dim=0).mean().detach().numpy())


# Reconstruction + KL divergence losses summed over all elements and batch
def cross_entropy_loss(logits, x, reduce_batch=True):
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    # shape (batch_size,)
    negative_log_p_theta = bce(input=logits, target=x.view(-1, 784)).sum(-1)
    if reduce_batch:
        negative_log_p_theta = negative_log_p_theta.mean()
    return negative_log_p_theta
    # return F.binary_cross_entropy(
    #     bernoulli_vars, input_x.view(-1, 784), size_average=False) / input_x.shape[0]

def kl_divergence(qy):
    # From Eric Jang's code ...
    # logits_py = tf.ones_like(logits_y) * 1./K
    # p_cat_y = OneHotCategorical(logits=logits_py)
    # q_cat_y = OneHotCategorical(logits=logits_y)
    # KL_qp = tf.contrib.distributions.kl(q_cat_y, p_cat_y)
    return torch.sum(qy * torch.log(qy * args.categorical_dim + EPS),
                     dim=-1).mean()


def train(model, images, optimizer, epoch, metrics):
    model.train()
    train_bce_loss = 0
    train_kl_loss = 0
    temp = args.temp
    assert len(images[0].size()) == 4
    total_images = len(images) * images[0].size(0)
    for batch_idx, data in enumerate(images):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp, args.hard)
        bce_loss = cross_entropy_loss(recon_batch, data)
        kl_loss = kl_divergence(qy)
        loss = bce_loss + kl_loss
        # loss = bce_loss + max(0, 10 * (1 - (epoch / 50 ))) * kl_loss
        loss.backward()
        train_bce_loss += bce_loss.item() * len(data)
        train_kl_loss += kl_loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), TEMP_MIN)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tMean highest prob: {:.2f}'.format(
                epoch, batch_idx * len(data), total_images,
                       100. * batch_idx / len(images),
                       loss.item(),
                       torch.max(qy, dim=-1)[0].mean().detach().numpy()))
    avg_kl_loss = train_bce_loss / total_images
    avg_bce_loss = train_kl_loss / total_images
    metrics['bce_loss'].append(train_bce_loss / total_images)
    metrics['kl_loss'].append(train_kl_loss / total_images)
    metrics['elbo'].append(avg_kl_loss + avg_bce_loss)
    print('====> Epoch: {} BCE loss: {:.4f} KL loss: {:.4f}'.format(
        epoch, avg_kl_loss, avg_bce_loss))


def test(model, images, epoch, metrics):
    model.eval()
    test_loss = 0
    temp = args.temp
    for i, data in enumerate(images):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        bce_loss = cross_entropy_loss(recon_batch, data)
        kl_loss = kl_divergence(qy)
        test_loss += (bce_loss.item() + kl_loss.item()) * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), TEMP_MIN)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat(
                [data[:n],
                torch.bernoulli(torch.sigmoid(recon_batch)).view(args.batch_size, 1, 28, 28)[:n]])
            save_image(
                comparison.data.cpu(),
                os.path.join(args.experiment_directory,
                            f'reconstruction_{epoch}.png'),
                nrow=n)
    assert len(images[0].size()) == 4
    total_images = len(images) * images[0].size(0)
    test_loss /= total_images
    metrics['test_elbo'].append(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def save_sampled_images(model, epoch, num_images=64):
    rows = num_images * args.latent_dim
    label = np.zeros((rows, args.categorical_dim), dtype=np.float32)
    label[range(rows), np.random.choice(args.categorical_dim, rows)] = 1
    label = np.reshape(label, [num_images, args.latent_dim, args.categorical_dim])
    sample = torch.from_numpy(label).view(num_images, args.latent_dim * args.categorical_dim)
    if args.cuda:
        sample = sample.cuda()
    sample = torch.bernoulli(torch.sigmoid(model.decode(sample))).cpu()
    # print(sample.min(), sample.max())
    save_image(sample.data.view(num_images, 1, 28, 28),
               os.path.join(args.experiment_directory,
                            f'sample_{epoch}.png'),)


def main():
    if not os.path.exists(args.experiment_directory):
        os.mkdir(args.experiment_directory)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_images = [torch.bernoulli(data) for (data, _) in train_loader]
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_images = [torch.bernoulli(data) for (data, _) in test_loader]
    model = VAE_gumbel(args.temp)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    metrics = collections.defaultdict(list)
    for epoch in range(1, args.epochs + 1):
        train(model, train_images, optimizer, epoch, metrics)
        test(model, test_images, epoch, metrics)
        save_sampled_images(model, epoch)
        record_negative_iwae_metrics(
            model, torch.cat(test_images, 0),
            iw=args.importance_weights, metrics=metrics)
        print('niwae:', np.squeeze(metrics['niwae'][-1]))
    for metric_name, values in metrics.items():
        plt.plot(np.asarray(values).reshape(-1),
                 marker='o', label=metric_name)
    plt.legend()
    plt.savefig(os.path.join(args.experiment_directory,
                             'metrics.png'))
    with open(os.path.join(
        args.experiment_directory, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == '__main__':
    main()
