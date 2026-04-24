from model import *
from utils import *
import numpy as np
import torch
import os,sys
import argparse

import warnings
warnings.filterwarnings("ignore")

def main(args):

    
    for dataset in ['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']: #'GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter',
        trained_cpi_model_path = f'/home/datahouse1/liujin/PreMOTA/regression_multitask/model_fintune_save/{dataset}/ratio_0.9batch128LR_1e-4random_0_esm2.pt'

        USE_CUDA = torch.cuda.is_available()
        device = torch.device(args.device if USE_CUDA else 'cpu')
        print(device)


        embedding_path = "/home/datahouse1/liujin/PreMOTA/dataset_reg_multitask/%s/%s/"%(dataset,args.protein_embedding)
        model = CPI_classification(device,
                                emb_size=args.emb_size,
                                max_length=2000,
                                dropout=args.dropout_global,
                                modulator_emb_dim=args.modulator_emb_dim,
                                ppi_emb_dim=args.ppi_emb_dim,
                                h_dim=args.h_dim,
                                n_heads=args.n_heads,
                                n_output=1)
        model.to(device)
        
        checkpoint = torch.load(trained_cpi_model_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        print(f'predicting on model:{trained_cpi_model_path};device:{device};')


        dataset_path = "/home/datahouse1/liujin/CPI_offtarget/dataset_reg_multitask/" + dataset + '/test/'
        # test_fold = eval(open(dataset_path+'valid_entries.txt','r').read())
        if not os.path.exists(embedding_path):
            print("No protein embedding files, please generate relevant embedding first!")
            exit(0)
        
        print("start predicting...")
        test_data = create_dataset_for_test(dataset_path,embedding_path)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate, num_workers=2)
        T_pK, S_pK, T_pAC50, S_pAC50 = predicting(model, device, test_loader)
        print("end predicting...")
    
        current_rmse_pK, current_mse_pK, current_pearson_pK, current_spearman_pK, current_r2_pK, current_rm2_pK, current_ci_pK, current_cindex_pK = performance_evaluation(T_pK, S_pK)
        current_rmse_pAC50, current_mse_pAC50, current_pearson_pAC50, current_spearman_pAC50, current_r2_pAC50, current_rm2_pAC50, current_ci_pAC50, current_cindex_pAC50 = performance_evaluation(T_pAC50, S_pAC50)
        current_rmse = (current_rmse_pK + current_rmse_pAC50) / 2
        current_pearson = (current_pearson_pK + current_pearson_pAC50) / 2
        current_spearman = (current_spearman_pK + current_spearman_pAC50) / 2
        current_r2 = (current_r2_pK + current_r2_pAC50) / 2
        current_rm2 = (current_rm2_pK + current_rm2_pAC50) / 2
        current_cindex = (current_cindex_pK + current_cindex_pAC50) / 2
        print("*******************************************")
        print('Now the dataset is:{}'.format(dataset))
        print('Test RMSE:\t{}'.format(current_rmse))
        print('Test Pearson:\t{}'.format(current_pearson))
        print('Test Spearman:\t{}'.format(current_spearman))
        print('Test R2:\t{}'.format(current_r2))
        print('Test RM2:\t{}'.format(current_rm2))
        print('Test CI_index:\t{}'.format(current_cindex))
        print("*******************************************")

        #保存预测结果
        np.save(f'/home/datahouse1/liujin/CPI_offtarget/regression_multitask_new/predict_result/{dataset}_T_pK.npy',T_pK)
        np.save(f'/home/datahouse1/liujin/CPI_offtarget/regression_multitask_new/predict_result/{dataset}_S_pK.npy',S_pK)
        np.save(f'/home/datahouse1/liujin/CPI_offtarget/regression_multitask_new/predict_result/{dataset}_T_pAC50.npy',T_pAC50)
        np.save(f'/home/datahouse1/liujin/CPI_offtarget/regression_multitask_new/predict_result/{dataset}_S_pAC50.npy',S_pAC50)

#['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:4',type=str,help='device id (0,1,2,3)')
    # parser.add_argument('--dataset', type=str, default='Kinase',choices=['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other'])
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100) #100
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dropout_global', type=float, default=0)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=25) #10
    parser.add_argument('--stop_epoch', type=int, default=0)
    parser.add_argument('--best_epoch', type=int, default=-1)
    parser.add_argument('--best_pearson', type=float, default=0)
    parser.add_argument('--last_epoch', type=int, default=1)
    parser.add_argument('--best_model', type=str, default=None)
    parser.add_argument('--modulator_emb_dim', type=int, default=256)
    parser.add_argument('--ppi_emb_dim', type=int, default=512)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--protein_embedding', type=str, default='esm2',choices=['onehot','tape', 'esm1b', 'esm2','esm2_1024'])
    parser.add_argument('--emb_size', type=int, default=1280,choices=[20, 768, 1280, 1280])
    args = parser.parse_args()
    
    main(args)

#nohup python -u test.py >train_reg.log 2>&1 &
#nohup python -u test.py >fintune_reg.log 2>&1 &