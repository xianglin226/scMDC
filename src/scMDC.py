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
from utils import cluster_acc, torch_PCA
from preprocess import read_dataset, normalize
import scanpy as sc

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if type=="encode" and i==len(layers)-1:
            break
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class scMultiCluster(nn.Module):
    def __init__(self, input_dim1, input_dim2, zencode_dim=[64,16], zdecode_dim=[16,64],
            encodeLayer1=[256, 128, 64], decodeLayer1=[64, 128, 256], encodeLayer2=[8], decodeLayer2=[8], 
            activation="relu", sigma1=2.5, sigma2=1., alpha=1., gamma1=.01, gamma2=.1, gamma3=0.001, cutoff = 0.3):
        super(scMultiCluster, self).__init__()
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.z_dim = zencode_dim[-1]
        self.encoder1 = buildNetwork([input_dim1]+encodeLayer1, type="encode", activation=activation)
        self.decoder1 = buildNetwork(decodeLayer1, type="decode", activation=activation)
        self.encoder2 = buildNetwork([input_dim2]+encodeLayer2, type="encode", activation=activation)
        self.decoder2 = buildNetwork(decodeLayer2, type="decode", activation=activation)
        self.latent_enc = buildNetwork([encodeLayer1[-1]+encodeLayer2[-1]]+zencode_dim, type="encode", activation=activation)
        self.latent_dec = buildNetwork(zdecode_dim+[encodeLayer1[-1]+encodeLayer2[-1]], type="encode", activation=activation)        
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.NBLoss = NBLoss()
        self.mse = nn.MSELoss()

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

    def forward(self, x1, x2):
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        
        pi1 = self.dec_pi1(h1_)

        h10 = self.encoder1(x1)
        h20 = self.encoder2(x2)
        combine_latent0 = torch.cat([h10, h20], dim=-1)
        z0 = self.latent_enc(combine_latent0)
        combine_latent0_ = self.latent_dec(z0)
        q = self.soft_assign(z0)
        num, lq = self.cal_latent(z0)
        return z0, q, num, lq, mean1, mean2, disp1, disp2, pi1, combine_latent0, combine_latent0_
        
    def forward_AE(self, x1, x2):
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        
        pi1 = self.dec_pi1(h1_)

        h10 = self.encoder1(x1)
        h20 = self.encoder2(x2)
        combine_latent0 = torch.cat([h10, h20], dim=-1)
        z0 = self.latent_enc(combine_latent0)
        combine_latent0_ = self.latent_dec(z0)
        num, lq = self.cal_latent(z0)
        return z0, num, lq, mean1, mean2, disp1, disp2, pi1, combine_latent0, combine_latent0_
        
    def encodeBatch(self, X1, X2, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            
        encoded = []
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch)
            inputs2 = Variable(x2batch)
            z,_,_,_,_,_,_,_,_,_ = self.forward_AE(inputs1, inputs2)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss
    
    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q))
        c2 = -torch.sum(p * torch.log(p))
        l = c1 - c2
        return l

    def pretrain_autoencoder(self, X1, X_raw1, sf1, X2, X_raw2, sf2, 
            batch_size=256, lr=0.005, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1), torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        counts = 0
        for epoch in range(epochs):
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).cuda()
                x_raw1_tensor = Variable(x_raw1_batch).cuda()
                sf1_tensor = Variable(sf1_batch).cuda()
                x2_tensor = Variable(x2_batch).cuda()
                x_raw2_tensor = Variable(x_raw2_batch).cuda()
                sf2_tensor = Variable(sf2_batch).cuda()
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, combine_latent0, combine_latent0_ = self.forward_AE(x1_tensor, x2_tensor)
                #recon_loss1 = self.mse(mean1_tensor, x1_tensor)
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                #recon_loss2 = self.mse(mean2_tensor, x2_tensor)
                recon_loss2 = self.NBLoss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, scale_factor=sf2_tensor)
                recon_loss_latent = self.mse(combine_latent0_, combine_latent0) * self.gamma2
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) * self.gamma3
                if counts > epochs * self.cutoff:
                   loss = recon_loss1 + recon_loss2 + recon_loss_latent + kl_loss
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}, NB loss:{:.4f}, latent MSE loss:{:.8f}, KL loss:{:.8f}'.format(
                   batch_idx+1, epoch+1, recon_loss1.item(), recon_loss2.item(), recon_loss_latent.item(), kl_loss.item()))
                else:
                   loss = recon_loss1 + recon_loss2
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}, NB loss:{:.4f}'.format(
                   batch_idx+1, epoch+1, recon_loss1.item(), recon_loss2.item()))
            counts +=1

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X1, X_raw1, sf1, X2, X_raw2, sf2, y=None, lr=.1, n_clusters = 4,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X1 = torch.tensor(X1).cuda()
        X_raw1 = torch.tensor(X_raw1).cuda()
        sf1 = torch.tensor(sf1).cuda()
        X2 = torch.tensor(X2).cuda()
        X_raw2 = torch.tensor(X_raw2).cuda()
        sf2 = torch.tensor(sf2).cuda()
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
             
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
                q = self.soft_assign(Zdata)
                p = self.target_distribution(q).data
                
                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))

                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'p': p,
                            'q': q,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
                
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
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs1 = Variable(x1_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                inputs2 = Variable(x2_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)
                target1 = Variable(pbatch)

                zbatch, qbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, combine_latent0, combine_latent0_ = self.forward(inputs1, inputs2)
                
                cluster_loss = self.cluster_loss(target1, qbatch)
                #recon_loss1 = self.mse(mean1_tensor, inputs1)
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                #recon_loss2 = self.mse(mean2_tensor, inputs2)
                recon_loss2 = self.NBLoss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, scale_factor=sfinputs2)
                recon_loss_latent = self.mse(combine_latent0_, combine_latent0)
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch)
                loss = recon_loss_latent * self.gamma2 + cluster_loss * self.gamma1 + kl_loss * self.gamma3 + recon_loss1 + recon_loss2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                recon_loss_latent_val += recon_loss_latent.data * len(inputs1)
                kl_loss_val += kl_loss.data * len(inputs1)
                train_loss = recon_loss1_val + recon_loss2_val + recon_loss_latent_val + cluster_loss_val + kl_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.8f ZINB Loss: %.4f NB Loss: %.4f Latent MSE Loss: %.4f KL Loss: %.4f" % (
               epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss1_val / num, recon_loss2_val / num, recon_loss_latent_val / num, kl_loss_val / num))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
