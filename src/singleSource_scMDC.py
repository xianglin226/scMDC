import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from utils import cluster_acc, torch_PCA
from pytorch_kmeans import kmeans, initialize, pairwise_distance, pairwise_cosine, kmeans_predict
from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA
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
            encodeLayer1=[256, 128, 64], decodeLayer1=[64, 128, 256], encodeLayer2=[32,8], decodeLayer2=[8,32], 
            activation="relu", sigma1=2.5, sigma2=1., alpha=1., gamma1=.01, gamma2=.001, gamma3 = .1, cutoff = 0.5):
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
        self.r_dim = encodeLayer1[-1]
        self.p_dim = encodeLayer2[-1]
        self.encoder1 = buildNetwork([input_dim1]+encodeLayer1, type="encode", activation=activation)
        self.decoder1 = buildNetwork(decodeLayer1, type="decode", activation=activation)
        self.encoder2 = buildNetwork([input_dim2]+encodeLayer2, type="encode", activation=activation)
        self.decoder2 = buildNetwork(decodeLayer2, type="decode", activation=activation)
        self.latent_enc = buildNetwork([encodeLayer1[-1]+encodeLayer2[-1]]+zencode_dim, type="encode", activation=activation)
        #self.latent_enc = buildNetwork([encodeLayer1[-1]+ input_dim2]+zencode_dim, type="encode", activation=activation)
        self.latent_dec = buildNetwork(zdecode_dim+[encodeLayer1[-1]+encodeLayer2[-1]], type="encode", activation=activation)        
        #self.latent_dec = buildNetwork(zdecode_dim+[encodeLayer1[-1]+input_dim2], type="encode", activation=activation)
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
        
    def rbf_dist(self, x, y):
        dist = torch.cdist(x,y)
        gamma = 1./x.shape[1]
        return torch.exp(-gamma * dist)
        
    def get_raw_stress_loss(self, h1, h2, dist):
        return torch.mean(torch.square(torch.cdist(h1, h2)-dist))

    def get_norm_stress_loss(self, h1, h2, dist, eps = 1e-6):
        return torch.sum(torch.square(torch.cdist(h1, h2) - dist)) / torch.sum(torch.square(dist) + eps)
        
    def mds_loss(self, Z, X1=None, X2=None, w1 = 0.5, w2 = 0.5):
        pair_num = np.int32(Z.shape[0]/2)
        h1, h2 = Z[:pair_num], Z[pair_num:]
        if w1 == 0:
           dist2 = torch.cdist(X2[:pair_num], X2[pair_num:])
           dist3 = dist2 * w2
        elif w2 ==0:
           X1 = torch_PCA(X1, k = 30)
           dist1 = torch.cdist(X1[:pair_num], X1[pair_num:])
           dist3 = dist1 * w1
        else:
           X1 = torch_PCA(X1, k = 30)
           dist1 = torch.cdist(X1[:pair_num], X1[pair_num:])
           dist2 = torch.cdist(X2[:pair_num], X2[pair_num:])
           dist3 = dist1 * w1 + dist2 * w2        
        mds_loss = self.get_raw_stress_loss(h1, h2, dist3)
        return mds_loss
     
    def target_distribution1(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def target_distribution2(self, q):
        p = ((q**2).t() / q.sum(0)).t()
        return (p.t() / p.sum(1)).t()

    def forward_rna(self, x1):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)     
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        pi1 = self.dec_pi1(h1_)
        q = self.soft_assign(h1)
        num, lq = self.cal_latent(h1)
        return h1, q, num, lq, mean1, disp1, pi1
    
    def forward_protein(self, x2):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        q = self.soft_assign(h2)
        num, lq = self.cal_latent(h2)
        return h2, q, num, lq, mean2, disp2
    
    def forward_rna_AE(self, x1):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        pi1 = self.dec_pi1(h1_)
        num, lq = self.cal_latent(h1)
        return h1, num, lq, mean1, disp1, pi1
        
    def forward_protein_AE(self, x2):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        num, lq = self.cal_latent(h2)
        return h2, num, lq, mean2, disp2
    
    def encodeBatch_rna(self, X1, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch)
            z,_,_,_,_,_ = self.forward_rna_AE(inputs1)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded
    
    def encodeBatch_protein(self, X2, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        self.eval()
        num = X2.shape[0]
        num_batch = int(math.ceil(1.0*X2.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs2 = Variable(x2batch)
            z,_,_,_,_ = self.forward_protein_AE(inputs2)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q))
        c2 = -torch.sum(p * torch.log(p))
        l = c1 - c2
        return l
        
    def pretrain_autoencoder_rna(self, X1, X_raw1, sf1,
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        pca1 = PCA(n_components=20)
        X1_pca = pca1.fit_transform(X1)
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            counts = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).cuda()
                x_raw1_tensor = Variable(x_raw1_batch).cuda()
                sf1_tensor = Variable(sf1_batch).cuda()
                zbatch, z_num, lqbatch, mean1_tensor, disp1_tensor, pi1_tensor = self.forward_rna_AE(x1_tensor)
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                #recon_loss2 = self.mse(mean2_tensor, x2_tensor)
                lpbatch = self.target_distribution1(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) * self.gamma2             
                loss = recon_loss1 + kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                counts = counts +1
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f} KL loss:{:.4f}'.format(
                batch_idx+1, epoch+1, recon_loss1.item(), kl_loss.item()))
        
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)
                    
    def pretrain_autoencoder_protein(self, X2, X_raw2, sf2, 
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X2.shape[0]/batch_size))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        adata2 = sc.AnnData(X_raw2)
        adata2 = read_dataset(adata2, transpose=False,test_split=False,copy=True)
        adata2 = normalize(adata2, size_factors=True,normalize_input=False,logtrans_input=True)
        X2_log = adata2.X
        pca2 = PCA(n_components=5)
        X2_pca = pca2.fit_transform(X2_log)
        dataset = TensorDataset(torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            counts = 0
            for batch_idx, (x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                x2_tensor = Variable(x2_batch).cuda()
                x_raw2_tensor = Variable(x_raw2_batch).cuda()
                sf2_tensor = Variable(sf2_batch).cuda()
                zbatch, z_num, lqbatch, mean2_tensor, disp2_tensor = self.forward_protein_AE(x2_tensor)
                recon_loss2 = self.NBLoss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, scale_factor=sf2_tensor)
                #recon_loss2 = self.mse(mean2_tensor, x2_tensor)
                lpbatch = self.target_distribution1(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) * self.gamma2
                loss = recon_loss2 + kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                counts = counts + 1
                print('Pretrain epoch [{}/{}], NB loss:{:.4f} KL loss:{:.4f}'.format(
                batch_idx+1, epoch+1, recon_loss2.item(), kl_loss.item()))
        
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit_RNA(self, X1, X_raw1, sf1, y=None, lr=1., n_clusters = 4,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        pca1 = PCA(n_components=20)
        X1_pca = pca1.fit_transform(X1)
        X1 = torch.tensor(X1).cuda()
        X1_pca = torch.tensor(X1_pca).cuda()
        X_raw1 = torch.tensor(X_raw1).cuda()
        sf1 = torch.tensor(sf1).cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001)
        
        print("Initializing cluster centers with kmeans.")
        self.mu = Parameter(torch.Tensor(n_clusters, self.r_dim))
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch_rna(X1, batch_size=batch_size)
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
                # update the targe distribution p
                Zdata = self.encodeBatch_rna(X1, batch_size=batch_size)     
                q = self.soft_assign(Zdata)
                p = self.target_distribution1(q).data
                self.y_pred = kmeans_predict(X=Zdata, cluster_centers = self.mu, distance='euclidean', device=torch.device('cuda:0'))
                self.y_pred = self.y_pred.data.cpu().numpy()
                
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
                x1_pca_batch = X1_pca[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs1 = Variable(x1_batch)
                pca1 = Variable(x1_pca_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                target1 = Variable(pbatch)
                
                zbatch, qbatch, z_num, lqbatch, mean1_tensor, disp1_tensor, pi1_tensor = self.forward_rna(inputs1)

                #cluster_loss = self.cluster_loss(target1, qbatch)
                dis = pairwise_distance(zbatch, self.mu, device=torch.device('cuda:0'))
                choice_cluster = torch.min(dis, dim=1)
                cluster_loss = torch.log(torch.sum(choice_cluster.values)) * self.gamma1
                #mds_loss = self.mds_loss(zbatch, X1 = pca1, w1 = self.w1, w2 = self.w2)
                target2 = self.target_distribution1(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch) * self.gamma2
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                loss = recon_loss1 + cluster_loss + kl_loss
                loss.backward()
                optimizer.step()
                kl_loss_val += kl_loss.data * len(inputs1)
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)
                train_loss = recon_loss1_val + cluster_loss_val + kl_loss_val
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f KL Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss1_val / num, kl_loss_val / num))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
        
        
    def fit_protein(self, X2, X_raw2, sf2, y=None, lr=1., n_clusters = 4,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        adata2 = sc.AnnData(X_raw2)
        adata2 = read_dataset(adata2, transpose=False,test_split=False,copy=True)
        adata2 = normalize(adata2, size_factors=True,normalize_input=False,logtrans_input=True)
        X2_log = adata2.X
        pca2 = PCA(n_components=5)
        X2_pca = pca2.fit_transform(X2_log)
        X2 = torch.tensor(X2).cuda()
        X2_pca = torch.tensor(X2_pca).cuda()
        X_raw2 = torch.tensor(X_raw2).cuda()
        sf2 = torch.tensor(sf2).cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001)

        print("Initializing cluster centers with kmeans.")
        self.mu = Parameter(torch.Tensor(n_clusters, self.p_dim))
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch_protein(X2, batch_size=batch_size)
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        self.train()
        num = X2.shape[0]
        num_batch = int(math.ceil(1.0*X2.shape[0]/batch_size))

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                Zdata = self.encodeBatch_protein(X2, batch_size=batch_size)
                q = self.soft_assign(Zdata)
                p = self.target_distribution1(q).data 
                self.y_pred = kmeans_predict(X=Zdata, cluster_centers = self.mu, distance='euclidean', device=torch.device('cuda:0'))
                self.y_pred = self.y_pred.data.cpu().numpy()

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
            #mds_loss_val = 0.0
            kl_loss_val = 0.0
            for batch_idx in range(num_batch):
                x2_batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x2_pca_batch = X2_pca[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf2_batch = sf2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs2 = Variable(x2_batch)
                pca2 = Variable(x2_pca_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)
                target1 = Variable(pbatch)

                zbatch, qbatch, z_num, lqbatch, mean2_tensor, disp2_tensor = self.forward_protein(inputs2)

                #cluster_loss = self.cluster_loss(target1, qbatch)
                dis = pairwise_distance(zbatch, self.mu, device=torch.device('cuda:0'))
                choice_cluster = torch.min(dis, dim=1)
                cluster_loss = torch.log(torch.sum(choice_cluster.values)) * self.gamma1
                #mds_loss = self.mds_loss(zbatch, X2 = pca2, w1 = self.w1, w2 = self.w2)
                target2 = self.target_distribution1(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch) * self.gamma2
                recon_loss2 = self.NBLoss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, scale_factor=sfinputs2)
                loss = recon_loss2 + cluster_loss + kl_loss
                loss.backward()
                optimizer.step()
                kl_loss_val += kl_loss.data * len(inputs2)
                #mds_loss_val += mds_loss.data * len(inputs2)
                cluster_loss_val += cluster_loss.data * len(inputs2)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                train_loss = recon_loss2_val + cluster_loss_val + kl_loss_val
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f NB Loss: %.4f KL Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss2_val / num, kl_loss_val / num))
        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
        
