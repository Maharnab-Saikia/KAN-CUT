import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
#from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_nce_loss = 0.0

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                
                for k, v in losses.items():
                    if k != 'NCE_List':
                        if k == 'G_GAN':
                            epoch_g_loss += v * opt.batch_size / dataset_size
                        if k == 'D':
                            epoch_d_loss += v * opt.batch_size / dataset_size
                        if k == 'NCE':
                            epoch_nce_loss += v * opt.batch_size / dataset_size
                
                
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            #model.save_networks(epoch)
            model.save_checkpoint(epoch)

        print(f"\n[Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}] Loss => G: {epoch_g_loss:.4f} | D: {epoch_d_loss:.4f} | NCE: {epoch_nce_loss:.4f} || Time Taken: {int(time.time() - epoch_start_time)} sec")
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
