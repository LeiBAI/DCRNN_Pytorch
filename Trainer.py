
import torch
import math
import os
import time
import matplotlib.pyplot as plt
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader,
                 args, logger, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                label = target[..., :self.model.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0)
                #output = torch.transpose(output.view(12, data.shape[0], self.model.num_nodes,self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)
                loss = self.loss(output.cuda(), label)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            label = target[..., :self.model.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            global_step = (epoch - 1) * self.train_per_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step,
                                                                     self.args.tf_decay_steps)
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            #output = torch.transpose(output.view(12, data.shape[0], self.model.num_nodes,self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)
            loss = self.loss(output.cuda(), label)
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.3f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.3f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
        #learning rate decay
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        #validation
        if self.val_loader == None:
            val_epoch_loss = 0
        else:
            val_epoch_loss = self.val_epoch(epoch)
        return train_epoch_loss, val_epoch_loss

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss, val_epoch_loss = self.train_epoch(epoch)
            print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e5:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if self.val_loader == None:
                val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if not_improved_count == self.args.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args.early_stop))
                break
            # save the best state
            if best_state == True:
                self.save_checkpoint()
            #plot loss figure
            if self.args.plot == True:
                self._plot_line_figure([train_loss_list, val_loss_list], path=self.loss_figure_path)
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format(training_time/60))

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                label = target[..., :model.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], 0.1, 0.1)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, 0.1, 0.1)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

    @staticmethod
    def _plot_line_figure(losses, path):
        train_loss = losses[0]
        val_loss = losses[1]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss)+1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(path, bbox_inches="tight")