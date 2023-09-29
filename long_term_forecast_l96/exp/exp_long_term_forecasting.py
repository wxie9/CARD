from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np

import wandb



warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,is_test = True):
        total_loss = []
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



                if self.args.model == 'CARD' and is_test == False:
                    ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to('cuda')
                    outputs = outputs * ratio
                    batch_y = batch_y * ratio



                pred = outputs#.detach().cpu()
                true = batch_y#.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item()*batch_y.shape[0])
                total_samples += batch_y.shape[0]
        total_loss = np.sum(total_loss) /total_samples
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test0')



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        c = nn.L1Loss()

        if self.args.lradj == 'TST':
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        else:
            scheduler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)




                    if self.args.model == 'CARD':
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs *self.ratio
                        batch_y = batch_y *self.ratio
                        loss = c(outputs, batch_y)




                        use_h_loss = False
                        h_level_range = [4,8,16,24,48,96]
                        h_loss = None
                        if use_h_loss:
                            
                            for h_level in h_level_range:
                                batch,length,channel = outputs.shape
                                # print(outputs.shape)
                                h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
                                h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
                                h_ratio = self.ratio[:h_outputs.shape[-2],:]
                                # print(h_outputs.shape,h_ratio.shape)
                                h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
                                h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)


                                h_outputs = h_outputs*h_ratio
                                h_batch_y = h_batch_y*h_ratio

                                h_ouputs_agg *= h_ratio
                                h_batch_y_agg *= h_ratio

                                if h_loss is None:
                                    h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                                else:
                                    h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                            # outputs = 0


                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if h_loss != 0:
                        loss = loss #+ h_loss * 1e-2
                    loss.backward()
                    model_optim.step()


                if self.args.lradj == 'TST':
                    adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)


            if self.args.model == 'CARD':
                vali_loss = self.vali(vali_data, vali_loader, c,is_test = False)
                test_loss = self.vali(test_data, test_loader, nn.MSELoss(),is_test = True)
            else:
                test_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            # test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({"Train Loss": train_loss," Vali Loss":vali_loss,"Test loss tmp": test_loss})
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break




            if self.args.lradj != 'TST': 
                adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        wandb.log({"test mae": mae," test mse":mse})
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# from torch.optim import lr_scheduler 
# import os
# import time
# import warnings
# import numpy as np



# warnings.filterwarnings('ignore')


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion,is_test = True,setting = None,epoch = 0):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


#                 # self.down_sample_ratio = 8

#                 # for i in range(2):
#                 h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                 # print(h_output_low.shape)
#                 h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
#                 h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


#                 h_output_high1 = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                     # print(h_output_tmp.shape)
#                 h_output_high1 = h_output_high1.reshape(h_output_high1.shape[0],h_output_high1.shape[1],h_output_high1.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
                
#                 # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
#                     # print(h_output_tmp.shape)
#                 h_outputs = h_output_low*(1-self.tmp00) + h_output_high1*self.tmp00
#                 h_output_high = h_output_high1 - h_output_low
#                 h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)


#                 if self.args.model == 'CARD' and is_test == False:
#                     self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                     self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
#                         # print(h_outputs.shape,self.ratio.shape,batch_y.shape)
#                         # input()
#                     h_outputs = h_outputs *self.ratio
#                     batch_y = batch_y *self.ratio

#                     self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
#                     self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
#                     batch_y_low = batch_y.transpose(-1,-2)
#                     batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                     batch_y_low = torch.mean(batch_y_low,dim = -1)

#                     # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

#                     h_output_low = h_output_low.squeeze(-1) *self.ratio_low
#                     batch_y_low = batch_y_low *self.ratio_low
#                     batch_y_high = batch_y.transpose(-1,-2)
#                     batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                     batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)

#                         # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

#                     h_output_low = h_output_low.squeeze(-1) *self.ratio_low
#                     batch_y_low = batch_y_low *self.ratio_low

#                     # loss = c(h_outputs, batch_y) + c(h_output_low, batch_y_low)



#                 pred = h_outputs#.detach().cpu()
#                 true = batch_y#.detach().cpu()
#                 # loss =    criterion(h_output_high, batch_y_high) + criterion(h_output_low, batch_y_low)
#                 loss = criterion(pred, true)# criterion(h_outputs, batch_y) +

#                 folder_path = './test_results/' + setting + '/'



#                 h_outputs = h_outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()


#                 # torch.Size([1, 7, 12, 8]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12, 8])
#                 # print(batch_y_low.shape)
#                 # torch.Size([128, 96, 7]) torch.Size([128, 96, 7])
#                 # print(batch_y_low.shape,batch_y_high.shape)
#                 batch_y_low = (batch_y_low.unsqueeze(-1) + 0 * batch_y_high).detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
#                 # print(batch_y_low.shape)
#                 # input()
#                 # print(h_output_low.shape)
#                 h_output_low = (h_output_low.unsqueeze(-1) + 0 * h_output_high).detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)

#                 # print(h_output_low.shape)
#                 batch_y_high = batch_y_high.detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
#                 h_output_high = h_output_high.detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
#                 # print(batch_x.shape,batch_y.shape,h_outputs.shape,batch_y_low.shape,h_output_low.shape,batch_y_high.shape,h_output_high.shape)
#                 # batch_x
#                 if i % 20 == 0 and i < 10:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], h_outputs[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + f'_{epoch}_.pdf'))

#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], batch_y_low[0, -1, :]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], h_output_low[0, -1, :]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path,  f'_low_{str(i)}_{epoch}.pdf'))


#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], batch_y_high[0, -1, :]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], h_output_high[0, -1, :]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path,  f'high_{str(i)}_{epoch}.pdf'))
#                 total_loss.append(loss.item())
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='val')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')



#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         c = nn.L1Loss()

#         if self.args.lradj == 'TST':
#             train_steps = len(train_loader)
#             scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
#                                             steps_per_epoch = train_steps,
#                                             pct_start = self.args.pct_start,
#                                             epochs = self.args.train_epochs,
#                                             max_lr = self.args.learning_rate)
#         else:
#             scheduler = None
#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)

#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


#                     self.down_sample_ratio = 8
#                     self.tmp00 = 1
#                     # for i in range(2):
#                     h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                     # print(h_output_low.shape)
#                     h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
#                     h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


#                     h_output_high = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                         # print(h_output_tmp.shape)
#                     h_output_high = h_output_high.reshape(h_output_high.shape[0],h_output_high.shape[1],h_output_high.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
#                     # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
#                         # print(h_output_tmp.shape)

                    
#                     h_outputs = h_output_low*(1-self.tmp00) + h_output_high*self.tmp00
#                     h_output_high = h_output_high - h_output_low
#                     h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)



#                     if self.args.model == 'CARD':
#                         self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                         self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
#                         # print(h_outputs.shape,self.ratio.shape,batch_y.shape)
#                         # input()
#                         h_outputs = h_outputs *self.ratio
#                         batch_y = batch_y *self.ratio

#                         self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
#                         self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
#                         batch_y_low = batch_y.transpose(-1,-2)
#                         batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                         batch_y_low = torch.mean(batch_y_low,dim = -1)

#                         # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

#                         h_output_low = h_output_low.squeeze(-1) *self.ratio_low
#                         batch_y_low = batch_y_low *self.ratio_low
#                         batch_y_high = batch_y.transpose(-1,-2)
#                         batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                         batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)
#                         # print(batch_y_high.shape,h_output_high.shape)
#                         # loss = c(h_outputs, batch_y) + c(h_output_low, batch_y_low)*1e3
#                         loss = c(h_outputs, batch_y) # + c(h_output_low, batch_y_low)#*1e-1# * 1e3#*1e3


#                     else:
#                         loss = criterion(outputs, batch_y)

#                     train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()


#                 if self.args.lradj == 'TST':
#                     adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
#                     scheduler.step()
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)


#             if self.args.model == 'CARD':
#                 vali_loss = self.vali(vali_data, vali_loader, c,is_test = False,setting = setting,epoch = epoch)
#             else:
#                 vali_loss = self.vali(vali_data, vali_loader, criterion)

#             print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
#             # test_loss = self.vali(test_data, test_loader, criterion)

#             # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#             #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break




#             if self.args.lradj != 'TST': 
#                 adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
#             else:
#                 print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

#                     else:
#                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



#                 # self.down_sample_ratio = 4

#                 # for i in range(2):
#                 h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                 # print(h_output_low.shape)
#                 h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
#                 h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


#                 h_output_high = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
#                     # print(h_output_tmp.shape)
#                 h_output_high = h_output_high.reshape(h_output_high.shape[0],h_output_high.shape[1],h_output_high.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
#                 # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
#                     # print(h_output_tmp.shape)
#                 h_outputs = h_output_low*(1-self.tmp00) + h_output_high*self.tmp00
#                 h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)
#                 h_output_high = h_output_high - h_output_low


#                 # self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
#                 # self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
#                 batch_y_low = batch_y.transpose(-1,-2)
#                 batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                 batch_y_low = torch.mean(batch_y_low,dim = -1)

#                 # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

#                 h_output_low = h_output_low.squeeze(-1)# *self.ratio_low
#                 batch_y_low = batch_y_low #*self.ratio_low
#                 batch_y_high = batch_y.transpose(-1,-2)
#                 batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
#                 batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)

#                 # print(batch_y_high.shape,batch_y_low.shape,h_output_low.shape,h_output_high.shape)

#                 h_outputs = h_outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()


#                 # torch.Size([1, 7, 12, 8]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12, 8])
#                 # print(batch_y_low.shape)
#                 batch_y_low = (batch_y_low.unsqueeze(-1) + 0 * batch_y_high).detach().cpu().numpy().reshape(1,7,-1)
#                 # print(batch_y_low.shape)
#                 # input()
#                 # print(h_output_low.shape)
#                 h_output_low = (h_output_low.unsqueeze(-1) + 0 * h_output_high).detach().cpu().numpy().reshape(1,7,-1)

#                 # print(h_output_low.shape)
#                 batch_y_high = batch_y_high.detach().cpu().numpy().reshape(1,7,-1)
#                 h_output_high = h_output_high.detach().cpu().numpy().reshape(1,7,-1)

                

#                 pred = h_outputs
#                 true = batch_y

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0 and i < 10:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], batch_y_low[0, -1, :]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], h_output_low[0, -1, :]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '_low.pdf'))


#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], batch_y_high[0, -1, :]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], h_output_high[0, -1, :]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '_high.pdf'))

#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         # np.save(folder_path + 'pred.npy', preds)
#         # np.save(folder_path + 'true.npy', trues)

#         return

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# from torch.optim import lr_scheduler 
# import os
# import time
# import warnings
# import numpy as np

# import wandb



# warnings.filterwarnings('ignore')


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag,dataset = None):
#         data_set, data_loader = data_provider(self.args, flag,dataset= dataset)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion,is_test = True):
#         total_loss = []
#         total_samples = 0
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



#                 if self.args.model == 'CARD' and is_test == False:
#                     ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                     ratio = torch.tensor(ratio).unsqueeze(-1).to('cuda')
#                     outputs = outputs * ratio
#                     batch_y = batch_y * ratio



#                 pred = outputs#.detach().cpu()
#                 true = batch_y#.detach().cpu()

#                 loss = criterion(pred, true)

#                 total_loss.append(loss.item()*batch_y.shape[0])
#                 total_samples += batch_y.shape[0]
#         total_loss = np.sum(total_loss) /total_samples
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test0')
        
        
#         # for data_path in ['ETTh1','ETTh2','ETTm1','ETTm2']:
#         self.args.data_path = 'ETTh1.csv'
#         train_data_h1, train_loader_h1 = self._get_data(flag='train',dataset ='ETTh1')
#         vali_data_h1, vali_loader_h1 = self._get_data(flag='val',dataset ='ETTh1')
#         test_data_h1, test_loader_h1 = self._get_data(flag='test0',dataset ='ETTh1')

#         self.args.data_path = 'ETTh2.csv'
#         train_data_h2, train_loader_h2 = self._get_data(flag='train',dataset ='ETTh2')
#         vali_data_h2, vali_loader_h2 = self._get_data(flag='val',dataset ='ETTh2')
#         test_data_h2, test_loader_h2 = self._get_data(flag='test0',dataset ='ETTh2')

#         self.args.data_path = 'ETTm1.csv'
#         train_data_m1, train_loader_m1 = self._get_data(flag='train',dataset ='ETTm1')
#         vali_data_m1, vali_loader_m1 = self._get_data(flag='val',dataset ='ETTm1')
#         test_data_m1, test_loader_m1 = self._get_data(flag='test0',dataset ='ETTm1')

#         self.args.data_path = 'ETTm2.csv'
#         train_data_m2, train_loader_m2 = self._get_data(flag='train',dataset ='ETTm2')
#         vali_data_m2, vali_loader_m2 = self._get_data(flag='val',dataset ='ETTm2')
#         test_data_m2, test_loader_m2 = self._get_data(flag='test0',dataset ='ETTm2')
#     # 'ETTh1': Dataset_ETT_hour,
#     # 'ETTh2': Dataset_ETT_hour,
#     # 'ETTm1': Dataset_ETT_minute,
#     # 'ETTm2': Dataset_ETT_minute,

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         c = nn.L1Loss()

#         if self.args.lradj == 'TST':
#             train_steps = len(train_loader)
#             scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
#                                             steps_per_epoch = train_steps,
#                                             pct_start = self.args.pct_start,
#                                             epochs = self.args.train_epochs,
#                                             max_lr = self.args.learning_rate)
#         else:
#             scheduler = None
#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         train_loader_list = [train_loader_h1]*4 + [train_loader_h2] * 4 + [train_loader_m1] * 4 + [train_loader_m2] * 4
#         vali_lodaer_list = [vali_loader_h1,vali_loader_h2,vali_loader_m1,vali_loader_m2]
#         test_lodaer_list = [test_loader_h1,test_loader_h2,test_loader_m1,test_loader_m2]
#         for epoch in range(self.args.train_epochs):
            
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for train_loader in train_loader_list:
#                 for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                     iter_count += 1
#                     model_optim.zero_grad()
#                     batch_x = batch_x.float().to(self.device)

#                     batch_y = batch_y.float().to(self.device)
#                     batch_x_mark = batch_x_mark.float().to(self.device)
#                     batch_y_mark = batch_y_mark.float().to(self.device)

#                     # decoder input
#                     dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                     dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                     # encoder - decoder
#                     if self.args.use_amp:
#                         with torch.cuda.amp.autocast():
#                             if self.args.output_attention:
#                                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                             else:
#                                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                             f_dim = -1 if self.args.features == 'MS' else 0
#                             outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                             loss = criterion(outputs, batch_y)
#                             train_loss.append(loss.item())
#                     else:
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)




#                         if self.args.model == 'CARD':
#                             self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                             self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
#                             outputs = outputs *self.ratio
#                             batch_y = batch_y *self.ratio
#                             loss = c(outputs, batch_y)




#                             use_h_loss = False
#                             h_level_range = [4,8,16,24,48,96]
#                             h_loss = None
#                             if use_h_loss:
                                
#                                 for h_level in h_level_range:
#                                     batch,length,channel = outputs.shape
#                                     # print(outputs.shape)
#                                     h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
#                                     h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
#                                     h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
#                                     h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
#                                     h_ratio = self.ratio[:h_outputs.shape[-2],:]
#                                     # print(h_outputs.shape,h_ratio.shape)
#                                     h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
#                                     h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)


#                                     h_outputs = h_outputs*h_ratio
#                                     h_batch_y = h_batch_y*h_ratio

#                                     h_ouputs_agg *= h_ratio
#                                     h_batch_y_agg *= h_ratio

#                                     if h_loss is None:
#                                         h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
#                                     else:
#                                         h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
#                                 # outputs = 0


#                         else:
#                             loss = criterion(outputs, batch_y)

#                         train_loss.append(loss.item())

#                     if (i + 1) % 100 == 0:
#                         print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                         speed = (time.time() - time_now) / iter_count
#                         left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                         print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                         iter_count = 0
#                         time_now = time.time()

#                     if self.args.use_amp:
#                         scaler.scale(loss).backward()
#                         scaler.step(model_optim)
#                         scaler.update()
#                     else:
#                         if h_loss != 0:
#                             loss = loss #+ h_loss * 1e-2
#                         loss.backward()
#                         model_optim.step()


#                     if self.args.lradj == 'TST':
#                         adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
#                         scheduler.step()
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)


#             if self.args.model == 'CARD':
#                 vali_loss_list = []
#                 test_loss_list = []
#                 for vali_loader in vali_lodaer_list:
#                     vali_loss_list.append(self.vali(vali_data, vali_loader, c,is_test = False))
#                 for test_loader in test_lodaer_list:
#                     test_loss_list.append(self.vali(test_data, test_loader, nn.MSELoss(),is_test = True))
#             else:
#                 test_loss = self.vali(vali_data, vali_loader, criterion)

#             for i in range(4):
#                 vali_loss = vali_loss_list[i]
#                 test_loss = test_loss_list[i]
#                 print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
#             # test_loss = self.vali(test_data, test_loader, criterion)
#             wandb.log({"Train Loss": train_loss," Vali Loss":vali_loss,"Test loss tmp": test_loss})
#             # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#             #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(np.mean(vali_loss_list), self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break




#             if self.args.lradj != 'TST': 
#                 adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
#             else:
#                 print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()

#                 pred = outputs
#                 true = batch_y

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)

#         wandb.log({"test mae": mae," test mse":mse})
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         # np.save(folder_path + 'pred.npy', preds)
#         # np.save(folder_path + 'true.npy', trues)

#         return

# # from data_provider.data_factory import data_provider
# # from exp.exp_basic import Exp_Basic
# # from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
# # from utils.metrics import metric
# # import torch
# # import torch.nn as nn
# # from torch import optim
# # from torch.optim import lr_scheduler 
# # import os
# # import time
# # import warnings
# # import numpy as np



# # warnings.filterwarnings('ignore')


# # class Exp_Long_Term_Forecast(Exp_Basic):
# #     def __init__(self, args):
# #         super(Exp_Long_Term_Forecast, self).__init__(args)

# #     def _build_model(self):
# #         model = self.model_dict[self.args.model].Model(self.args).float()

# #         if self.args.use_multi_gpu and self.args.use_gpu:
# #             model = nn.DataParallel(model, device_ids=self.args.device_ids)
# #         return model

# #     def _get_data(self, flag):
# #         data_set, data_loader = data_provider(self.args, flag)
# #         return data_set, data_loader

# #     def _select_optimizer(self):
# #         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
# #         return model_optim

# #     def _select_criterion(self):
# #         criterion = nn.MSELoss()
# #         return criterion

# #     def vali(self, vali_data, vali_loader, criterion,is_test = True,setting = None,epoch = 0):
# #         total_loss = []
# #         self.model.eval()
# #         with torch.no_grad():
# #             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
# #                 batch_x = batch_x.float().to(self.device)
# #                 batch_y = batch_y.float()

# #                 batch_x_mark = batch_x_mark.float().to(self.device)
# #                 batch_y_mark = batch_y_mark.float().to(self.device)

# #                 # decoder input
# #                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
# #                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
# #                 # encoder - decoder
# #                 if self.args.use_amp:
# #                     with torch.cuda.amp.autocast():
# #                         if self.args.output_attention:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
# #                         else:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
# #                 else:
# #                     if self.args.output_attention:
# #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
# #                     else:
# #                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
# #                 f_dim = -1 if self.args.features == 'MS' else 0
# #                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
# #                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


# #                 # self.down_sample_ratio = 8

# #                 # for i in range(2):
# #                 h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                 # print(h_output_low.shape)
# #                 h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
# #                 h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


# #                 h_output_high1 = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                     # print(h_output_tmp.shape)
# #                 h_output_high1 = h_output_high1.reshape(h_output_high1.shape[0],h_output_high1.shape[1],h_output_high1.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
                
# #                 # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
# #                     # print(h_output_tmp.shape)
# #                 h_outputs = h_output_low*(1-self.tmp00) + h_output_high1*self.tmp00
# #                 h_output_high = h_output_high1 - h_output_low
# #                 h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)


# #                 if self.args.model == 'CARD' and is_test == False:
# #                     self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
# #                     self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
# #                         # print(h_outputs.shape,self.ratio.shape,batch_y.shape)
# #                         # input()
# #                     h_outputs = h_outputs *self.ratio
# #                     batch_y = batch_y *self.ratio

# #                     self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
# #                     self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
# #                     batch_y_low = batch_y.transpose(-1,-2)
# #                     batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                     batch_y_low = torch.mean(batch_y_low,dim = -1)

# #                     # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

# #                     h_output_low = h_output_low.squeeze(-1) *self.ratio_low
# #                     batch_y_low = batch_y_low *self.ratio_low
# #                     batch_y_high = batch_y.transpose(-1,-2)
# #                     batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                     batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)

# #                         # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

# #                     h_output_low = h_output_low.squeeze(-1) *self.ratio_low
# #                     batch_y_low = batch_y_low *self.ratio_low

# #                     # loss = c(h_outputs, batch_y) + c(h_output_low, batch_y_low)



# #                 pred = h_outputs#.detach().cpu()
# #                 true = batch_y#.detach().cpu()
# #                 # loss =    criterion(h_output_high, batch_y_high) + criterion(h_output_low, batch_y_low)
# #                 loss = criterion(pred, true)# criterion(h_outputs, batch_y) +

# #                 folder_path = './test_results/' + setting + '/'



# #                 h_outputs = h_outputs.detach().cpu().numpy()
# #                 batch_y = batch_y.detach().cpu().numpy()


# #                 # torch.Size([1, 7, 12, 8]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12, 8])
# #                 # print(batch_y_low.shape)
# #                 # torch.Size([128, 96, 7]) torch.Size([128, 96, 7])
# #                 # print(batch_y_low.shape,batch_y_high.shape)
# #                 batch_y_low = (batch_y_low.unsqueeze(-1) + 0 * batch_y_high).detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
# #                 # print(batch_y_low.shape)
# #                 # input()
# #                 # print(h_output_low.shape)
# #                 h_output_low = (h_output_low.unsqueeze(-1) + 0 * h_output_high).detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)

# #                 # print(h_output_low.shape)
# #                 batch_y_high = batch_y_high.detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
# #                 h_output_high = h_output_high.detach().cpu().numpy().reshape(batch_y_high.shape[0],7,-1)
# #                 # print(batch_x.shape,batch_y.shape,h_outputs.shape,batch_y_low.shape,h_output_low.shape,batch_y_high.shape,h_output_high.shape)
# #                 # batch_x
# #                 if i % 20 == 0 and i < 10:
# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], h_outputs[0, :, -1]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path, str(i) + f'_{epoch}_.pdf'))

# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], batch_y_low[0, -1, :]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], h_output_low[0, -1, :]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path,  f'_low_{str(i)}_{epoch}.pdf'))


# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], batch_y_high[0, -1, :]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], h_output_high[0, -1, :]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path,  f'high_{str(i)}_{epoch}.pdf'))
# #                 total_loss.append(loss.item())
# #         total_loss = np.average(total_loss)
# #         self.model.train()
# #         return total_loss

# #     def train(self, setting):
# #         train_data, train_loader = self._get_data(flag='val')
# #         vali_data, vali_loader = self._get_data(flag='val')
# #         test_data, test_loader = self._get_data(flag='test')



# #         path = os.path.join(self.args.checkpoints, setting)
# #         if not os.path.exists(path):
# #             os.makedirs(path)

# #         time_now = time.time()

# #         train_steps = len(train_loader)
# #         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

# #         model_optim = self._select_optimizer()
# #         criterion = self._select_criterion()

# #         c = nn.L1Loss()

# #         if self.args.lradj == 'TST':
# #             train_steps = len(train_loader)
# #             scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
# #                                             steps_per_epoch = train_steps,
# #                                             pct_start = self.args.pct_start,
# #                                             epochs = self.args.train_epochs,
# #                                             max_lr = self.args.learning_rate)
# #         else:
# #             scheduler = None
# #         if self.args.use_amp:
# #             scaler = torch.cuda.amp.GradScaler()

# #         for epoch in range(self.args.train_epochs):
# #             iter_count = 0
# #             train_loss = []

# #             self.model.train()
# #             epoch_time = time.time()
# #             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
# #                 iter_count += 1
# #                 model_optim.zero_grad()
# #                 batch_x = batch_x.float().to(self.device)

# #                 batch_y = batch_y.float().to(self.device)
# #                 batch_x_mark = batch_x_mark.float().to(self.device)
# #                 batch_y_mark = batch_y_mark.float().to(self.device)

# #                 # decoder input
# #                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
# #                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

# #                 # encoder - decoder
# #                 if self.args.use_amp:
# #                     with torch.cuda.amp.autocast():
# #                         if self.args.output_attention:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
# #                         else:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# #                         f_dim = -1 if self.args.features == 'MS' else 0
# #                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
# #                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
# #                         loss = criterion(outputs, batch_y)
# #                         train_loss.append(loss.item())
# #                 else:
# #                     if self.args.output_attention:
# #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
# #                     else:
# #                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# #                     f_dim = -1 if self.args.features == 'MS' else 0
# #                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
# #                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


# #                     self.down_sample_ratio = 8
# #                     self.tmp00 = 1
# #                     # for i in range(2):
# #                     h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                     # print(h_output_low.shape)
# #                     h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
# #                     h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


# #                     h_output_high = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                         # print(h_output_tmp.shape)
# #                     h_output_high = h_output_high.reshape(h_output_high.shape[0],h_output_high.shape[1],h_output_high.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
# #                     # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
# #                         # print(h_output_tmp.shape)

                    
# #                     h_outputs = h_output_low*(1-self.tmp00) + h_output_high*self.tmp00
# #                     h_output_high = h_output_high - h_output_low
# #                     h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)



# #                     if self.args.model == 'CARD':
# #                         self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
# #                         self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
# #                         # print(h_outputs.shape,self.ratio.shape,batch_y.shape)
# #                         # input()
# #                         h_outputs = h_outputs *self.ratio
# #                         batch_y = batch_y *self.ratio

# #                         self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
# #                         self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
# #                         batch_y_low = batch_y.transpose(-1,-2)
# #                         batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                         batch_y_low = torch.mean(batch_y_low,dim = -1)

# #                         # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

# #                         h_output_low = h_output_low.squeeze(-1) *self.ratio_low
# #                         batch_y_low = batch_y_low *self.ratio_low
# #                         batch_y_high = batch_y.transpose(-1,-2)
# #                         batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                         batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)
# #                         # print(batch_y_high.shape,h_output_high.shape)
# #                         # loss = c(h_outputs, batch_y) + c(h_output_low, batch_y_low)*1e3
# #                         loss = c(h_outputs, batch_y) # + c(h_output_low, batch_y_low)#*1e-1# * 1e3#*1e3


# #                     else:
# #                         loss = criterion(outputs, batch_y)

# #                     train_loss.append(loss.item())

# #                 if (i + 1) % 100 == 0:
# #                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
# #                     speed = (time.time() - time_now) / iter_count
# #                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
# #                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
# #                     iter_count = 0
# #                     time_now = time.time()

# #                 if self.args.use_amp:
# #                     scaler.scale(loss).backward()
# #                     scaler.step(model_optim)
# #                     scaler.update()
# #                 else:
# #                     loss.backward()
# #                     model_optim.step()


# #                 if self.args.lradj == 'TST':
# #                     adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
# #                     scheduler.step()
# #             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
# #             train_loss = np.average(train_loss)


# #             if self.args.model == 'CARD':
# #                 vali_loss = self.vali(vali_data, vali_loader, c,is_test = False,setting = setting,epoch = epoch)
# #             else:
# #                 vali_loss = self.vali(vali_data, vali_loader, criterion)

# #             print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
# #             # test_loss = self.vali(test_data, test_loader, criterion)

# #             # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
# #             #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
# #             early_stopping(vali_loss, self.model, path)
# #             if early_stopping.early_stop:
# #                 print("Early stopping")
# #                 break




# #             if self.args.lradj != 'TST': 
# #                 adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
# #             else:
# #                 print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


# #         best_model_path = path + '/' + 'checkpoint.pth'
# #         self.model.load_state_dict(torch.load(best_model_path))

# #         return self.model

# #     def test(self, setting, test=0):
# #         test_data, test_loader = self._get_data(flag='test')
# #         if test:
# #             print('loading model')
# #             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

# #         preds = []
# #         trues = []
# #         folder_path = './test_results/' + setting + '/'
# #         if not os.path.exists(folder_path):
# #             os.makedirs(folder_path)

# #         self.model.eval()
# #         with torch.no_grad():
# #             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
# #                 batch_x = batch_x.float().to(self.device)
# #                 batch_y = batch_y.float().to(self.device)

# #                 batch_x_mark = batch_x_mark.float().to(self.device)
# #                 batch_y_mark = batch_y_mark.float().to(self.device)

# #                 # decoder input
# #                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
# #                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
# #                 # encoder - decoder
# #                 if self.args.use_amp:
# #                     with torch.cuda.amp.autocast():
# #                         if self.args.output_attention:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
# #                         else:
# #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
# #                 else:
# #                     if self.args.output_attention:
# #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

# #                     else:
# #                         outputs,h_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# #                 f_dim = -1 if self.args.features == 'MS' else 0
# #                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
# #                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



# #                 # self.down_sample_ratio = 4

# #                 # for i in range(2):
# #                 h_output_low = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                 # print(h_output_low.shape)
# #                 h_output_low = h_output_low.reshape(h_output_low.shape[0],h_output_low.shape[1],h_output_low.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
# #                 h_output_low = torch.mean(h_output_low,dim = -1,keepdims = True)


# #                 h_output_high = h_outputs[1].transpose(-1,-2)[:,-self.args.pred_len:, f_dim:].transpose(-1,-2)
# #                     # print(h_output_tmp.shape)
# #                 h_output_high = h_output_high.reshape(h_output_high.shape[0],h_output_high.shape[1],h_output_high.shape[2] // self.down_sample_ratio,self.down_sample_ratio)
# #                 # h_output_high = torch.mean(h_output_high,dim = -1,keepdims = True)
# #                     # print(h_output_tmp.shape)
# #                 h_outputs = h_output_low*(1-self.tmp00) + h_output_high*self.tmp00
# #                 h_outputs = h_outputs.reshape(h_outputs.shape[0],h_outputs.shape[1],-1).transpose(-1,-2)
# #                 h_output_high = h_output_high - h_output_low


# #                 # self.ratio_low = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len // self.down_sample_ratio)])
# #                 # self.ratio_low = torch.tensor(self.ratio_low).unsqueeze(0).to('cuda')
# #                 batch_y_low = batch_y.transpose(-1,-2)
# #                 batch_y_low = batch_y_low.reshape(batch_y_low.shape[0],batch_y_low.shape[1],batch_y_low.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                 batch_y_low = torch.mean(batch_y_low,dim = -1)

# #                 # print(batch_y_low.shape,h_output_low.shape,self.ratio_low.shape )

# #                 h_output_low = h_output_low.squeeze(-1)# *self.ratio_low
# #                 batch_y_low = batch_y_low #*self.ratio_low
# #                 batch_y_high = batch_y.transpose(-1,-2)
# #                 batch_y_high = batch_y_high.reshape(batch_y_high.shape[0],batch_y_high.shape[1],batch_y_high.shape[2] // self.down_sample_ratio, self.down_sample_ratio)
# #                 batch_y_high = batch_y_high - batch_y_low.unsqueeze(-1)

# #                 # print(batch_y_high.shape,batch_y_low.shape,h_output_low.shape,h_output_high.shape)

# #                 h_outputs = h_outputs.detach().cpu().numpy()
# #                 batch_y = batch_y.detach().cpu().numpy()


# #                 # torch.Size([1, 7, 12, 8]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12]) torch.Size([1, 7, 12, 8])
# #                 # print(batch_y_low.shape)
# #                 batch_y_low = (batch_y_low.unsqueeze(-1) + 0 * batch_y_high).detach().cpu().numpy().reshape(1,7,-1)
# #                 # print(batch_y_low.shape)
# #                 # input()
# #                 # print(h_output_low.shape)
# #                 h_output_low = (h_output_low.unsqueeze(-1) + 0 * h_output_high).detach().cpu().numpy().reshape(1,7,-1)

# #                 # print(h_output_low.shape)
# #                 batch_y_high = batch_y_high.detach().cpu().numpy().reshape(1,7,-1)
# #                 h_output_high = h_output_high.detach().cpu().numpy().reshape(1,7,-1)

                

# #                 pred = h_outputs
# #                 true = batch_y

# #                 preds.append(pred)
# #                 trues.append(true)
# #                 if i % 20 == 0 and i < 10:
# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], batch_y_low[0, -1, :]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], h_output_low[0, -1, :]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path, str(i) + '_low.pdf'))


# #                     input = batch_x.detach().cpu().numpy()
# #                     gt = np.concatenate((input[0, :, -1], batch_y_high[0, -1, :]), axis=0)
# #                     pd = np.concatenate((input[0, :, -1], h_output_high[0, -1, :]), axis=0)
# #                     visual(gt, pd, os.path.join(folder_path, str(i) + '_high.pdf'))

# #         preds = np.array(preds)
# #         trues = np.array(trues)
# #         print('test shape:', preds.shape, trues.shape)
# #         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
# #         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
# #         print('test shape:', preds.shape, trues.shape)

# #         # result save
# #         folder_path = './results/' + setting + '/'
# #         if not os.path.exists(folder_path):
# #             os.makedirs(folder_path)

# #         mae, mse, rmse, mape, mspe = metric(preds, trues)
# #         print('mse:{}, mae:{}'.format(mse, mae))
# #         f = open("result_long_term_forecast.txt", 'a')
# #         f.write(setting + "  \n")
# #         f.write('mse:{}, mae:{}'.format(mse, mae))
# #         f.write('\n')
# #         f.write('\n')
# #         f.close()

# #         # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
# #         # np.save(folder_path + 'pred.npy', preds)
# #         # np.save(folder_path + 'true.npy', trues)

# #         return
