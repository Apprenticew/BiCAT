import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support


class evaluating_indicator():
    def __init__(self, class_num=4):

        self.class_num = class_num
        self.topk_accuracy = 0
        self.f1_score = 0
        self.confusion_matrix = torch.zeros((class_num, class_num))
        self.TN, self.TP, self.FN, self.FP = torch.zeros((class_num)), \
            torch.zeros((class_num)), torch.zeros((class_num)), torch.zeros((class_num))
        self.k_acc = {}
        self.iters = 0

    def update_confusion_matrix(self, output, target):
        '''
        update confusion matrix by model prediction scores and targets in one iter/batch
        
        INPUT
        -------------------------
        output: pred score in shape of (m,)
        target: ground truth in shape of (m,)
        '''

        if not torch.equal(output, output.floor()):
            _, pred = torch.max(output, dim=1)
            # threshold = 0.95
            # max_prob, max_idx = output[:, :3].max(dim=1)
            # pred = torch.where(output[:, 3] > threshold, torch.tensor(3), max_idx)
        else:
            pred = output.int()

        for (t, p) in zip(target, pred):
            self.confusion_matrix[t, p] += 1

    def summerize_confusion_matrix(self, f1_type='macro', weights=None):
        '''
        
        summerize indicators like F1-score from confusion matrix
        
        '''

        assert f1_type in ['macro', 'weighted'], 'f1_type must in ["marco","weighted"]'
        micro_eps = 1e-8
        TN, TP, FN, FP = torch.zeros((self.class_num)), torch.zeros((self.class_num)), \
            torch.zeros((self.class_num)), torch.zeros((self.class_num))
        Precision, Recall, F1, Accuracy, Specificity, NPV = torch.zeros((self.class_num)), torch.zeros(
            (self.class_num)), \
            torch.zeros((self.class_num)), torch.zeros((self.class_num)), torch.zeros((self.class_num)), torch.zeros(
            (self.class_num))

        for i in range(self.class_num):
            TP[i] = self.confusion_matrix[i, i].float()
            FP[i] = self.confusion_matrix[:, i].sum() - TP[i]
            FN[i] = self.confusion_matrix[i, :].sum() - TP[i]
            TN[i] = self.confusion_matrix[:, :].sum() - TP[i] - FP[i] - FN[i]
            Precision[i] = TP[i] / (TP[i] + FP[i] + micro_eps)
            Recall[i] = TP[i] / (TP[i] + FN[i] + micro_eps)
            Accuracy[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
            Specificity[i] = TN[i] / (TN[i] + FP[i])
            NPV[i] = TN[i] / (TN[i] + FN[i] + micro_eps)
            F1[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i] + micro_eps)

        F1_summary = None
        if f1_type == 'macro':
            F1_summary = F1.mean()
        elif f1_type == 'weighted':
            # assert weights, 'weights must not be empty'
            F1_summary = F1.dot(weights / weights.sum())

        # return {'TP':TP, 'FP':FP,'TN':TN,'FN':FN,'Precision':Precision,'Recall':Recall,\
        #          'F1_summary':F1_summary, 'F1':F1}
        return {'Precision': Precision, 'Recall': Recall, 'Accuracy': Accuracy, 'Specificity': Specificity, 'NPV': NPV,
                'F1_summary': F1_summary, 'F1': F1}

    def calculate_accuracy(self, output, target, topk=(1,)):

        """
        calculate top-k accuracy over the predictions for the specified value k
        
        INPUT
        -------------------------
        output: predictions scores(m,n) for a batch of data 
        target: groud truth(m,) of a batch of data
        top_k: a tuple of k for needed top_k accuracy
        
        RETURN
        -------------------------
        res: a list of items for average accuracy of a batch data
        
        """

        self.iters += 1
        with torch.no_grad():
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            # print(pred.device)

            batch_size = output.size(0)
            # print(target.device,pred.device)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:, :k].float().sum(dim=1)
                correct_k_avg_batch = correct_k.sum(dim=0).mul_(100 / batch_size)
                res.append(correct_k_avg_batch)

            return res

    def update_accuracy(self, output, target, topk=(1,)):
        '''
        accumulate accuracy of every batch
        '''

        if self.iters == 0:
            for k in topk:
                self.k_acc[k] = 0
        res = self.calculate_accuracy(output, target, topk)
        for (k, acc_val) in zip(topk, res):
            self.k_acc[k] += acc_val

    def summerize_accuracy(self):
        '''
         summarize accuracy of every batch
        '''

        for k, acc_val in self.k_acc.items():
            sum_acc_val = acc_val / self.iters
            self.k_acc[k] = sum_acc_val

# output = torch.rand((5,4))
# target = torch.tensor([1,2,0,1,2],dtype = int)  
# print(output)      
# print(calculate_accuracy(output,target,(1,2)))


# def weights_init(m):
#     """
#     Initialise weights of the model.
#     """
#     if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif (type(m) == nn.BatchNorm2d):
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


# class NormalNLLLoss:
#     """
#     Calculate the negative log likelihood
#     of normal distribution.
#     This needs to be minimised.

#     Treating Q(cj | x) as a factored Gaussian.


#     """

#     def __call__(self, x, mu, var):

#         logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
#             (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
#         nll = -(logli.sum(1).mean())

#         return nll


# def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device, labels=None, supervised=True):
#     """
#     Sample random noise vector for training.

#     INPUT
#    ------------------------
#     n_dis_c : Number of discrete latent code.
#     dis_c_dim : Dimension of discrete latent code.
#     n_con_c : Number of continuous latent code.
#     n_z : Dimension of incompressible noise.
#     batch_size : Batch Size
#     device : GPU/CPU
#     labels : Tensor of labels for supervised discrete latent

#     OUTPUT
#     ------------------------
#     noise : Sampled noise with shape (bsize,n_z+n_dis_c*dis_c_dim+n_con_c)
#     idx : class label 0/1 with shape (n_dis_c,bsize) , in MNIST, n_dis_c=10
#     """

#     z = torch.randn(batch_size, n_z, 1, 1, device=device)
#     idx = np.zeros((n_dis_c, batch_size))

#     if (n_dis_c != 0):
#         dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

#         for i in range(n_dis_c):

#             if supervised:
#                 idx[i] = labels.cpu().numpy()
#             else:
#                 idx[i] = np.random.randint(dis_c_dim, size=batch_size)

#             dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

#         dis_c = dis_c.view(batch_size, -1, 1, 1)

#     if (n_con_c != 0):
#         # Random uniform between -1 and 1.
#         con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

#     noise = z
#     if (n_dis_c != 0):
#         noise = torch.cat((z, dis_c), dim=1)
#     if (n_con_c != 0):
#         noise = torch.cat((noise, con_c), dim=1)

#     return noise, idx


# def cal_gradient(discriminator, netD, real_data, fake_data, batch_size, lambda_gp=10.0,device='cuda',info=False):

#     alpha = torch.rand(batch_size, 1, 1, 1).uniform_(0, 1)
#     alpha = alpha.expand(batch_size, *real_data.shape[1:]).to(device)
#     interp = alpha * real_data+(1 - alpha) * fake_data
#     if info:
#         pred_disc_interp = netD(discriminator(interp))
#     else:
#         pred_disc_interp = discriminator(interp)
#     gradients = torch.autograd.grad(
#         outputs=pred_disc_interp,
#         inputs=interp,
#         grad_outputs=torch.ones(*pred_disc_interp.shape).to(device),
#         retain_graph=True,  # ensure graph be retained after grad() and used in backward()
#         create_graph=True,  # ?
#     )[0]
#     gradients = gradients.view(batch_size, -1)
#     gradients_gp = lambda_gp*(((gradients+1e-16).norm(2, dim=1)-1)**2).mean()

#     return gradients,gradients_gp
