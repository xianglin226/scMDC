from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
import numpy as np

import math, os

from utils import torch_PCA

from preprocess import read_dataset, normalize
import scanpy as sc

def buildNetwork1(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if type=="encode" and i==len(layers)-1:
            break
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

def buildNetwork2(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="selu":
            net.append(nn.SELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

class scMultiClusterBatch(nn.Module):
    def __init__(self, input_dim1, input_dim2, n_batch,
            encodeLayer=[], decodeLayer1=[], decodeLayer2=[], tau=1., t=10, device = "cuda",
            activation="elu", sigma1=2.5, sigma2=.1, alpha=1., gamma=1., phi1=0.0001, phi2=0.0001, cutoff = 0.5):
        super(scMultiClusterBatch, self).__init__()
        self.tau=tau
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.t=t
        self.device = device
        self.encoder = buildNetwork2([input_dim1+input_dim2+n_batch]+encodeLayer, type="encode", activation=activation)
        self.decoder1 = buildNetwork2([decodeLayer1[0]+n_batch]+decodeLayer1[1:], type="decode", activation=activation)
        self.decoder2 = buildNetwork2([decodeLayer2[0]+n_batch]+decodeLayer2[1:], type="decode", activation=activation)       
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.NBLoss = NBLoss()
        self.mse = nn.MSELoss()
        self.z_dim = encodeLayer[-1]

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p
     
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
        
    def kmeans_loss(self, z):
        dist1 = self.tau * torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))
        
    def forward(self, x1, x2, b):
        x = torch.cat([x1+torch.randn_like(x1)*self.sigma1, x2+torch.randn_like(x2)*self.sigma2], dim=-1)
        h = self.encoder(torch.cat([x, b], dim=-1))
        h = torch.cat([h, b], dim=-1)

        h1 = self.decoder1(h)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)

        h2 = self.decoder2(h)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)

        x0 = torch.cat([x1, x2], dim=-1)
        h0 = self.encoder(torch.cat([x0, b], dim=-1))
        q = self.soft_assign(h0)
        num, lq = self.cal_latent(h0)
        return h0, q, num, lq, mean1, mean2, disp1, disp2, pi1, pi2

    def forwardAE(self, x1, x2, b):
        x = torch.cat([x1+torch.randn_like(x1)*self.sigma1, x2+torch.randn_like(x2)*self.sigma2], dim=-1)
        h = self.encoder(torch.cat([x, b], dim=-1))
        h = torch.cat([h, b], dim=-1)

        h1 = self.decoder1(h)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)
        
        h2 = self.decoder2(h)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)

        x0 = torch.cat([x1, x2], dim=-1)
        h0 = self.encoder(torch.cat([x0, b], dim=-1))
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean1, mean2, disp1, disp2, pi1, pi2
        
    def encodeBatch(self, X1, X2, B, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        encoded = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            b_batch = B[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch).to(self.device)
            inputs2 = Variable(x2batch).to(self.device)
            b_tensor = Variable(b_batch).to(self.device)
            z,_,_,_,_,_,_,_,_ = self.forwardAE(inputs1.float(), inputs2.float(), b_tensor.float())
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)

    def SDis_func(self, x, y):
        return torch.sum(torch.square(x - y), dim=1)

    def pretrain_autoencoder(self, X1, X_raw1, sf1, X2, X_raw2, sf2, B,
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1), torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2), torch.Tensor(B))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        counts = 0
        for epoch in range(epochs):
            loss_val = 0
            recon_loss1_val = 0
            recon_loss2_val = 0
            kl_loss_val = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch, b_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).to(self.device)
                x_raw1_tensor = Variable(x_raw1_batch).to(self.device)
                sf1_tensor = Variable(sf1_batch).to(self.device)
                x2_tensor = Variable(x2_batch).to(self.device)
                x_raw2_tensor = Variable(x_raw2_batch).to(self.device)
                sf2_tensor = Variable(sf2_batch).to(self.device)
                b_tensor = Variable(b_batch).to(self.device)
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor = self.forwardAE(x1_tensor, x2_tensor, b_tensor)
                #recon_loss1 = self.mse(mean1_tensor, x1_tensor)
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                #recon_loss2 = self.mse(mean2_tensor, x2_tensor)
                recon_loss2 = self.zinb_loss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_tensor)
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) 
                if epoch+1 >= epochs * self.cutoff:
                   loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1
                else:
                   loss = recon_loss1 + recon_loss2 #+ kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item() * len(x1_batch)
                recon_loss1_val += recon_loss1.item() * len(x1_batch)
                recon_loss2_val += recon_loss2.item() * len(x1_batch)
                if epoch+1 >= epochs * self.cutoff:
                    kl_loss_val += kl_loss.item() * len(x1_batch)

            loss_val = loss_val/X1.shape[0]
            recon_loss1_val = loss_val/X1.shape[0]
            recon_loss2_val = recon_loss2_val/X1.shape[0]
            kl_loss_val = kl_loss_val/X1.shape[0]
            if epoch%self.t == 0:
               print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss:{:.6f}, NB loss:{:.6f}, KL loss:{:.6f}'.format(epoch+1, loss_val, recon_loss1_val, recon_loss2_val, kl_loss_val))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X1, X_raw1, sf1, X2, X_raw2, sf2, B, y=None, lr=1., n_clusters = 4,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.to(self.device)
        print("Clustering stage")
        X1 = torch.tensor(X1).to(self.device)
        X_raw1 = torch.tensor(X_raw1).to(self.device)
        sf1 = torch.tensor(sf1).to(self.device)
        X2 = torch.tensor(X2).to(self.device)
        X_raw2 = torch.tensor(X_raw2).to(self.device)
        sf2 = torch.tensor(sf2).to(self.device)
        B = torch.tensor(B).to(self.device)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)
             
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch(X1, X2, B, batch_size=batch_size)
        #latent
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari))
        
        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))

        final_nmi, final_ari, final_epoch = 0, 0, 0

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                Zdata = self.encodeBatch(X1, X2, B, batch_size=batch_size)
                
                # evalute the clustering performance
                dist, _ = self.kmeans_loss(Zdata)
                self.y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()

                if y is not None:
                    #acc2 = np.round(cluster_acc(y, self.y_pred), 5)
                    final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    final_epoch = epoch+1
                    print('Clustering   %d: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, ami, nmi, ari))

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
                
                # save current model
                # if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    # self.save_checkpoint({'epoch': epoch+1,
                            # 'state_dict': self.state_dict(),
                            # 'mu': self.mu,
                            # 'y_pred': self.y_pred,
                            # 'y_pred_last': self.y_pred_last,
                            # 'y': y
                            # }, epoch+1, filename=save_dir)
                
            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            recon_loss_latent_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x2_batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf2_batch = sf2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                b_batch = B[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs1 = Variable(x1_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                inputs2 = Variable(x2_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)

                zbatch, qbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor = self.forward(inputs1.float(), inputs2.float(), b_batch.float())
                
                _, cluster_loss = self.kmeans_loss(zbatch)
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                recon_loss2 = self.zinb_loss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sfinputs2)
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch)
                loss = cluster_loss * self.gamma + kl_loss * self.phi2 + recon_loss1 + recon_loss2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                kl_loss_val += kl_loss.data * len(inputs1)
                loss_val = cluster_loss_val + recon_loss1_val + recon_loss2_val + kl_loss_val

            if epoch%self.t == 0:
                print("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss: %.6f ZINB Loss2: %.6f KL Loss: %.6f" % (
                     epoch + 1, loss_val / num, cluster_loss_val / num, recon_loss1_val / num, recon_loss2_val / num, kl_loss_val / num))

        return self.y_pred, final_epoch
