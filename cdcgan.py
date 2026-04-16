
"""
Based on: https://github.com/togheppi/cDCGAN/blob/master/Mame_cDCGAN_pytorch.py
"""

# Mame image generation using Conditional DCGAN
import torch
import config
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import imageio
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from io import BytesIO  # Python 3.x
from dataset import MAMeDataset
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        self.writer=SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        if isinstance(images, np.ndarray):
            images=torch.from_numpy(images)
        self.writer.add_images(tag, images, step)

    def histo_summary(self, tag, values, step, bins=1000):
        if isinstance(values, np.ndarray):
            values=torch.from_numpy(values)
        self.writer.add_histogram(tag, values, step, bins='auto')
# class Logger(object):
#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer=SummaryWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         # summary=tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         # self.writer.add_summary(summary, step)
#         self.writer.add_scalar(tag, value, step)

#     def image_summary(self, tag, images, step):
#         """Log a list of images."""

#         img_summaries=[]
#         for i, img in enumerate(images):
#             # Write the image to a string
#             s=BytesIO()
#             # scipy.misc.toimage(img).save(s, format="png")
#             plt.imsave(s, img, format='png')

#             # Create an Image object
#             img_sum=tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                     height=img.shape[0],
#                                     width=img.shape[1])
#             # Create a Summary value
#             img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

#         # Create and write Summary
#         summary=tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)

#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""

#         # Create a histogram using numpy
#         counts, bin_edges=np.histogram(values, bins=bins)

#         # Fill the fields of the histogram proto
#         hist=tf.HistogramProto()
#         hist.min=float(np.min(values))
#         hist.max=float(np.max(values))
#         hist.num=int(np.prod(values.shape))
#         hist.sum=float(np.sum(values))
#         hist.sum_squares=float(np.sum(values ** 2))

#         # Drop the start of the first bin
#         bin_edges=bin_edges[1:]

#         # Add bin edges and counts
#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)

#         # Create and write Summary
#         summary=tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()


# Parameters
image_size=256
label_dim=29
G_input_dim=100
G_output_dim=3
D_input_dim=3
D_output_dim=1
# num_filters=[1024, 512, 256, 128]
num_filters=[2048, 1024, 512, 256, 128, 64]

learning_rate=0.0002
betas=(0.5, 0.999)
batch_size=32
num_epochs=70

transform=transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

mamedata=MAMeDataset(root_dir=config.TRAIN_PATH, transform=transform, path_csv=config.CSV_PATH, classes_to_ind=None, num_imgs=0)

data_loader=torch.utils.data.DataLoader(dataset=mamedata,
                                        batch_size=batch_size,
                                        shuffle=True)


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out=(x + 1) / 2
    return out.clamp(0, 1)


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1=torch.nn.Sequential()
        self.hidden_layer2=torch.nn.Sequential()
        self.hidden_layer=torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i==0:
                # For input
                input_deconv=torch.nn.ConvTranspose2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv=torch.nn.ConvTranspose2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv=torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                deconv_name='deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)

                # Batch normalization
                bn_name='bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name='act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer=torch.nn.Sequential()
        # Deconvolutional layer
        out=torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, z, c):
        h1=self.hidden_layer1(z)
        h2=self.hidden_layer2(c)
        x=torch.cat([h1, h2], 1)
        h=self.hidden_layer(x)
        out=self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer1=torch.nn.Sequential()
        self.hidden_layer2=torch.nn.Sequential()
        self.hidden_layer=torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i==0:
                # For input
                input_conv=torch.nn.Conv2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv=torch.nn.Conv2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv=torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                conv_name='conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normalization
                bn_name='bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name='act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer=torch.nn.Sequential()
        # Convolutional layer
        out=torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        #self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z, c):
        h1=self.hidden_layer1(z)
        h2=self.hidden_layer2(c)
        x=torch.cat([h1, h2], 1)
        h=self.hidden_layer(x)
        out=self.output_layer(h)
        return out


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='Mame_cDCGAN_results/', show=False):
    fig, ax=plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn=save_dir + 'Mame_cDCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, label, num_epoch, save=False, save_dir='Mame_cDCGAN_results/', show=False, fig_size=(5, 5)):
    generator.eval()

    noise=Variable(noise.cuda())
    label=Variable(label.cuda())
    gen_image=generator(noise, label)
    gen_image=denorm(gen_image)

    generator.train()

    n_rows=np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols=np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes=plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img=(((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        # ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title='Epoch {0}'.format(num_epoch + 1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn=save_dir + 'Mame_cDCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_morp_result(generator, save=False, save_dir='Mame_cDCGAN_results/', show=False, fig_size=(10, 10)):
    source_z_=torch.randn(10, G_input_dim)
    z_=torch.zeros(10*10, G_input_dim)
    for i in range(5):
        for j in range(10):
            z_[i * 20 + j]=(source_z_[i * 2 + 1] - source_z_[i * 2]) / 9 * (j + 1) + source_z_[i * 2]

    for i in range(5):
        z_[i * 20 + 10:i * 20 + 20]=z_[i * 20:i * 20 + 10]

    y_=torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
    y_=torch.cat([y_, y_, y_, y_, y_], 0)
    y_label_=onehot[y_]
    noise=z_.view(-1, G_input_dim, 1, 1)
    label=y_label_.view(-1, label_dim, 1, 1)

    generator.eval()

    noise=Variable(noise.cuda())
    label=Variable(label.cuda())
    gen_image=generator(noise, label)
    gen_image=denorm(gen_image)

    generator.train()

    n_rows=np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols=np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes=plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img=(((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn=save_dir + 'Mame_cDCGAN_noise_morp.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Models
G=Generator(G_input_dim, label_dim, num_filters, G_output_dim)
D=Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)
G.cuda()
D.cuda()
save_dir='cDCGAN/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# Set the logger
D_log_dir=save_dir + 'D_logs'
G_log_dir=save_dir + 'G_logs'
if not os.path.exists(D_log_dir):
    os.mkdir(D_log_dir)
D_logger=Logger(D_log_dir)

if not os.path.exists(G_log_dir):
    os.mkdir(G_log_dir)
G_logger=Logger(G_log_dir)


# Loss function
# criterion=torch.nn.BCELoss()
criterion=torch.nn.BCEWithLogitsLoss()

# Optimizers
G_optimizer=torch.optim.Adam(G.parameters(), lr=2e-4, betas=betas)
D_optimizer=torch.optim.Adam(D.parameters(), lr=2e-4, betas=betas)


# Label preprocess
onehot=torch.zeros(label_dim, label_dim)
onehot=onehot.scatter_(1, torch.arange(label_dim).view(label_dim, 1), 1)
onehot=onehot.view(label_dim, label_dim, 1, 1)
fill=torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :]=1

# fixed noise & label
temp_noise0_=torch.randn(4, G_input_dim)
temp_noise0_=torch.cat([temp_noise0_, temp_noise0_], 0)
temp_noise1_=torch.randn(4, G_input_dim)
temp_noise1_=torch.cat([temp_noise1_, temp_noise1_], 0)

# fixed_noise=torch.cat([temp_noise0_, temp_noise1_], 0)
# fixed_label=torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()
fixed_classes=torch.arange(16) % label_dim
fixed_label=onehot[fixed_classes]
fixed_noise=torch.randn(16, G_input_dim).view(-1, G_input_dim, 1, 1)


# Training GAN
D_avg_losses=[]
G_avg_losses=[]

step=0
for epoch in range(num_epochs):
    D_losses=[]
    G_losses=[]

    # if epoch==5 or epoch==10:
    #     G_optimizer.param_groups[0]['lr'] /=10
    #     D_optimizer.param_groups[0]['lr'] /=10

    # minibatch training
    for i, (images, labels) in enumerate(data_loader):

        # image data
        mini_batch=images.size()[0]
        x_=Variable(images.cuda())

        # labels
        # y_real_=Variable(torch.ones(mini_batch).cuda())
        # y_fake_=Variable(torch.zeros(mini_batch).cuda())
        # c_label_=label[mini_batch*i:mini_batch*(i+1)]
        # c_fill_=Variable(fill[c_label_].cuda())
        c_label_=labels.long().view(-1)
        c_fill_=Variable(fill[c_label_].cuda())

        # Train discriminator with real data
        D_real_decision=D(x_, c_fill_)
        y_real_=torch.ones_like(D_real_decision)
        D_real_loss=criterion(D_real_decision, y_real_)

        # Train discriminator with fake data
        z_=torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        z_=Variable(z_.cuda())

        # c_=(torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        c_=torch.randint(0, label_dim, (mini_batch,), dtype=torch.long)
        c_onehot_=Variable(onehot[c_].cuda())
        gen_image=G(z_, c_onehot_).detach()

        c_fill_=Variable(fill[c_].cuda())
        D_fake_decision=D(gen_image, c_fill_)
        y_fake_=torch.zeros_like(D_fake_decision)
        D_fake_loss=criterion(D_fake_decision, y_fake_)

        # Back propagation
        D_loss=D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        z_=torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        z_=Variable(z_.cuda())

        c_=(torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        # c_=torch.randint(0, label_dim, (mini_batch,), device='cuda')
        c_onehot_=Variable(onehot[c_].cuda())
        gen_image=G(z_, c_onehot_)

        c_fill_=Variable(fill[c_].cuda())
        D_fake_decision=D(gen_image, c_fill_)
        y_real_=torch.ones_like(D_fake_decision)
        G_loss=criterion(D_fake_decision, y_real_)

        # Back propagation
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
            % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.item(), G_loss.item()))

        #============TensorBoard logging============#
        D_logger.scalar_summary('losses', D_loss.item(), step + 1)
        G_logger.scalar_summary('losses', G_loss.item(), step + 1)
        step +=1

    D_avg_loss=torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss=torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True, save_dir=save_dir)

    # Show result for fixed noise
    plot_result(G, fixed_noise, fixed_label, epoch, save=True, save_dir=save_dir)

def generate_augmented_dataset(generator, num_classes=29, images_per_class=100, save_root="augmented_cdcgan"):
    generator.eval()
    os.makedirs(save_root, exist_ok=True)
    with torch.no_grad():
        for class_ind in range(num_classes):
            class_dir=os.path.join(save_root, f"class_{class_ind}")
            os.makedirs(class_dir, exist_ok=True)
            for img_ind in range(images_per_class):
                z=torch.randn(1, G_input_dim, 1, 1).cuda()
                c=onehot[torch.tensor([class_ind])].cuda()
                generated=generator(z, c)
                img=denorm(generated)
                save_path=os.path.join(class_dir, f"class{class_ind}_{img_ind}.png")
                save_image(img, save_path)
    generator.train()


# Make gif
loss_plots=[]
gen_image_plots=[]
for epoch in range(num_epochs):
    # plot for generating gif
    save_fn1=save_dir + 'Mame_cDCGAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.imread(save_fn1))

    save_fn2=save_dir + 'Mame_cDCGAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.imread(save_fn2))
generate_augmented_dataset(generator=G, num_classes=label_dim, images_per_class=100, save_root="augmented_cdcgan")
imageio.mimsave(save_dir + 'Mame_cDCGAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
imageio.mimsave(save_dir + 'Mame_cDCGAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)

# plot noise morp result
# plot_morp_result(G, save=True, save_dir=save_dir)