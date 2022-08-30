import os
import numpy as np
import logging
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import CIFAR100
import torch.nn.functional as F
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import cvxpy as cvx
import scipy.io as scio
from torch.utils.data import Dataset,DataLoader,TensorDataset
import sys


class Net_f(nn.Module):
    def __init__(self):
        super(Net_f, self).__init__()
        googlenet = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.feature=torch.nn.Sequential(*list(googlenet.children())[0:18])
        self.fc1 = nn.Linear(1024,32)
        self.fc2 = nn.Linear(32,10)
        self.BN = nn.BatchNorm1d(10)

    def forward(self,x):
        out=self.feature(x)
        out=out.view(-1,1024)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        out=self.BN(out)

        return out      

class Net_g(nn.Module):
    def __init__(self,num_class=2, dim=10):
        super(Net_g, self).__init__()

        self.fc=nn.Linear(num_class, dim)

    def forward(self,x):
        out=self.fc(x)

        return out

def corr(f,g):
    k = torch.mean(torch.sum(f*g,1))
    return k
    
def cov_trace(f,g):
    cov_f = torch.mm(torch.t(f),f) / (f.size()[0]-1.)
    cov_g = torch.mm(torch.t(g),g) / (g.size()[0]-1.)
    return torch.trace(torch.mm(cov_f, cov_g))

# transform
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
    ])

def load_data(id, batch_size=100, t=0):
    '''
    task_id    range(21)
    batch_size 100
    t          0(train)/1(test)
    '''
    if t==0:
        x = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_train.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_train.npy"))
    else:
        x = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_test.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_test.npy"))
    data = torch.utils.data.DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return data

def train(id):
    sample, label = next(sourceiter[id - 1])
    labels_one_hot = torch.zeros(len(label), 2).scatter_(1, label.view(-1,1), 1)
    f = model_f(Variable(sample).to(device))
    g = model_g(Variable(labels_one_hot).to(device))

    loss = alpha0[id].item()*(-2) * corr(f, g)
    return loss

def get_fg(model_f, model_g, id): 
    dataset = load_data(id, 100)
    dataiter = iter(dataset)
    x, y = next(dataiter)
    y = torch.zeros(len(y), 2).scatter_(1, y.view(-1,1), 1)
    feature_f = model_f(Variable(x).to(device)).data.cpu().numpy()
    f = feature_f - np.sum(feature_f, 0) / feature_f.shape[0]
    feature_g = model_g(Variable(y).to(device)).data.cpu().numpy()
    g = feature_g - np.sum(feature_g, 0) / feature_g.shape[0]
    return f, g

def get_phicov(fx, gx, i):

    cov = (fx[i].T @ fx[i]) / fx[i].shape[0]
    covinv = np.linalg.inv(cov)

    phi = np.trace(((fx[0].T @ fx[i]) / fx[0].shape[0]) @ ((gx[0].T @ gx[i]) / gx[0].shape[0]))

    phi_cov = 0
    nd = 0
    for m in range(fx[0].shape[0]):
        for n in range(gx[0].shape[0]):
            if fx[i][m].T @ gx[i][n] > -1:
                phi_cov += (fx[0][m].T @ gx[0][n])*(fx[0][m].T @ gx[0][n]) * (1 + (fx[i][m].T @ gx[i][n]))
                nd += (fx[0][m].T @ gx[0][n]) * (1 + (fx[i][m].T @ gx[i][n]))
    phi_cov = phi_cov / fx[0].shape[0] / gx[0].shape[0] - ((nd / fx[0].shape[0] / gx[0].shape[0])**2)

    print(phi, '\t', phi_cov)
    
    return phi, phi_cov

def alpha(phi_x, phi_cov_x, dim, alpha0):
    "get optimized alpha from numerical solution"
    try:
        A = np.zeros((dim, dim))


        for i in range(1, dim):
            for j in range(1, dim):
                A[i][j] = (phi_x[i] - phi_x[0]) * (phi_x[j] - phi_x[0]) / phi_x[0]

        print(A)
        # A[0][0] = 1/(dim+1)*phi_cov_x[0] / phi_x[0]
        for i in range(dim):
            A[i][i] += phi_cov_x[i] / phi_x[0]
        # # !
        # B = np.ones((dim + 1, dim + 1))
        # B[:dim, :dim] = A * 2 
        # C = np.zeros((dim + 1, 1))
        # C[dim][0] = 1
        # D = np.linalg.inv(B) @ C
        # print(D)
        # A = A*10

        alphav=cvx.Variable(dim)
        obj = cvx.Minimize(cvx.quad_form(alphav, A))
        constraint = [np.ones([1,dim]) @ alphav == 1., np.eye(dim) @ alphav >= np.zeros(dim)]
        prob = cvx.Problem(obj, constraint)
        prob.solve() 
    
    except:
        logging.warning("-------fail to update alpha------")
        print("-------fail to update alpha------")
        return alpha0

    else:
        logging.info("-------update alpha finished------")
        print("-------update alpha finished------")
        return alphav.value


if __name__ == "__main__":
    time_start=time.time()

    log_path = './log/'
    logtime = time.strftime('%m%d_%H%M_%S_')

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(log_path, logtime + 'train.log'), level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    logging.info("available device: {}".format(device))

    DATA_PATH = 'data_set_2/'
    N_TASK = 8

    # alpha0 = torch.tensor([0.050390117, 0.047139142, 0.045838752, 0.051040312, 0.046163849, 0.047789337, 0.046814044, 0.045838752, 0.047789337, 0.045513654, 0.048439532, 0.048114434, 0.049089727, 0.047789337, 0.046814044, 0.045838752, 0.046488947, 0.047464239, 0.048114434, 0.049089727, 0.048439532])
    alpha0=torch.rand(N_TASK)
    epoch_out = 20
    
    for eo in range(epoch_out):
        print("=================epoch #{}================".format(eo))

        print('==========start training F and G==========')
        # ================= training process of F and G ======================
        target = load_data(0)
        testset = load_data(0, t=1)
        targetiter = iter(target)
        samplest, labelst=next(targetiter)
        labels_one_hot_t = torch.zeros(len(labelst), 2).scatter_(1, labelst.view(-1,1), 1)


        lr = 0.00001
        epoch_in = 10
        # alpha0 = torch.tensor([0.050390117, 0.047139142, 0.045838752, 0.051040312, 0.046163849, 0.047789337, 0.046814044, 0.045838752, 0.047789337, 0.045513654, 0.048439532, 0.048114434, 0.049089727, 0.047789337, 0.046814044, 0.045838752, 0.046488947, 0.047464239, 0.048114434, 0.049089727, 0.048439532])
        
        model_f = Net_f().to(device)
        model_g = Net_g().to(device)
        optimizer_fg = torch.optim.Adam(list(model_f.parameters()) + list(model_g.parameters()), lr = lr)
        
        losslist = []
        acclist = [0]

        for ei in range(epoch_in):
            
            sourceiter = []
            for id in range(1,N_TASK):
                source = load_data(id, batch_size=25)
                sourceiter.append(iter(source))

            losscc=[]
            for k in range(len(source)): 

                model_f.train()
                model_g.train()
                optimizer_fg.zero_grad()

                ft = model_f(Variable(samplest).to(device))
                gt = model_g(Variable(labels_one_hot_t).to(device))
                loss = alpha0[0].item()*(-2) * corr(ft, gt)
                for i in range(1,N_TASK):
                    loss += train(i)

                loss += 2 * ((torch.sum(ft,0)/ft.size()[0]) * (torch.sum(gt, 0) / gt.size()[0])).sum()
                loss += cov_trace(ft, gt)


                losscc.append(loss.item())
                loss.backward()
                optimizer_fg.step()


                model_f.eval()
                model_g.eval()

                acc=0
                total=0

                fc = model_f(Variable(samplest).to(device)).data.cpu().numpy()
                f_mean = np.sum(fc, axis = 0) / fc.shape[0]
                labellist = torch.Tensor([[1, 0], [0, 1]])
                gc = model_g(Variable(labellist).to(device)).data.cpu().numpy()
                gce = np.sum(gc,axis = 0) / gc.shape[0]
                gcp = gc - gce

                for k, data in enumerate(testset, 0):
                    samples, labels = data
                    labels = labels.numpy()
                    fc = model_f(Variable(samples).to(device)).data.cpu().numpy()
                    fcp = fc-f_mean
                    fgp = np.dot(fcp,gcp.T)
                    acc += (np.argmax(fgp, axis = 1) == labels).sum()
                    total += len(samples)

                acc = float(acc) / total
                print(acc)
                # if acc > 0.7:
                if acc > (max(acclist)):
                    print('changepara')
                    finalacc = acc
                    paraf = model_f.state_dict()
                    parag = model_g.state_dict()
                acclist.append(acc)

            per_ls = sum(losscc) / len(losscc)
            losslist.append(per_ls)
            print(per_ls)
            logging.info('epoch_out: {}, epoch_in: {}, loss: {}'.format(eo, ei, per_ls))
            print("--------------logging--------------")

            
        logging.info('The epoch_out: {}, epoch_in: {} of training F and G finished, losslist: {}, acclist: {}, finalacc: {}, alpha: {}'.format(eo, ei, losslist, acclist, finalacc, alpha0))
        #-----start loop------------
        print(losslist)
        print(acclist)
        print(finalacc)
        print(alpha0)

        torch.save(paraf, 'mpara/cifar100f_alpha_all.pth')
        torch.save(parag, 'mpara/cifar100g_alpha_all.pth')

        
        # ==============================================================

        # ====================== update alpha ==========================

        print("==============start training alpha=============")
        DIM = 8
        N = str(1)
        phi_x = []
        phi_cov_x = []

        model_f.load_state_dict(torch.load('mpara/cifar100f_alpha_all.pth', map_location='cpu'))
        model_g.load_state_dict(torch.load('mpara/cifar100g_alpha_all.pth', map_location='cpu'))
        model_f.eval()
        model_g.eval()

        feature_f, feature_g = [], [] 
        phi_x = []
        phi_cov_x = []

        for i in range(DIM):
            
            f, g = get_fg(model_f, model_g, i)
            feature_f.append(f)
            feature_g.append(g)

        for i in range(DIM):
            
            p, p_cov = get_phicov(feature_f, feature_g, i) 
            phi_x.append(p)
            phi_cov_x.append(p_cov)

        # dataset = load_data(0)
        # dataiter = iter(dataset)
        # data, label = next(dataiter)
        # label = torch.zeros(len(label), 2).scatter_(1, label.view(-1,1), 1)



        # feature_f = model_f(Variable(data).to(device)).data.cpu().numpy()
        # f = feature_f - np.sum(feature_f, 0) / feature_f.shape[0]
        # feature_g = model_g(Variable(label).to(device)).data.cpu().numpy()
        # g = feature_g - np.sum(feature_g, 0) / feature_g.shape[0]

        # feature_f_x, feature_g_x = [], [] 
        # for i in range(DIM):
            
        #     f, g = get_fg(i, data, label)
        #     feature_f_x.append(f)
        #     feature_g_x.append(g)                                                       

        # for i in range(DIM):
        #     p, p_cov = get_phicov(feature_f, feature_g, feature_f_x, feature_g_x, i) 
        #     phi_x.append(p)
        #     phi_cov_x.append(p_cov)

        # phi=np.trace(((feature_f.T @ feature_f)/feature_f.shape[0])@((feature_g.T @ feature_g)/feature_g.shape[0]))
        alpha_m = alpha(phi_x, phi_cov_x, DIM, alpha0)
        # ============================================================

        alpha0 = alpha_m
        logging.info('The epoch_out: {}, epoch_in: {} of training alpha finished, alpha: {}'.format(eo, ei, alpha0))

    time_end=time.time()
    print(time_end-time_start)
    logging.info('training time: {}'.format(time_end-time_start))


