import os, argparse, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

from scipy.ndimage import gaussian_filter
from models.unet import UNet
from utils.gen_mask import gen_mask
from losses.gms_loss import MSGMS_Score
from losses.hsv_loss import HSV_Score
from dataset.mvtec import MVTecDataset
import utils.funcs as f
import kornia

import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--data_type', type=str, default='50_euro/real_back')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/models/eur50_real_back/seed_4420/') #     3514     4420
    parser.add_argument('--save_dir', type=str, default='results/testing')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=np.random.randint(1, 10000))
    parser.add_argument('--k_value', type=int, nargs='+', default=[4, 8, 16, 32])
    args = parser.parse_args()

    # load model and dataset
    model = UNet().to(device)
    checkpoint = torch.load(args.checkpoint_dir + 'model.pt')
    model.load_state_dict(checkpoint['model'])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MVTecDataset(args.data_path, class_name=args.data_type, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    scores_MSGMS, scores_HSV, test_imgs, recon_imgs, gt_list, gt_mask_list = test(args, model, test_loader)

    print(f'Seed: {args.seed}')

    scores_MSGMS = f.normalize_scores(scores_MSGMS)
    #scores_H = f.normalize_scores(scores_H)
    #scores_S = f.normalize_scores(scores_S)
    #scores_V = f.normalize_scores(scores_V)
    #scores_H = np.asarray(scores_H)


    # calculate weight maps
    #f.calculate_score_weight_map(scores_MSGMS, os.path.join(args.checkpoint_dir, 'weight_map_MSGMS.npy'))
    #f.calculate_score_weight_map(scores_H, os.path.join(args.checkpoint_dir, 'weight_map_H.npy'))
    #f.calculate_score_weight_map(scores_H, os.path.join(args.checkpoint_dir, 'weight_map_S.npy'))
    #f.calculate_score_weight_map(scores_V, os.path.join(args.checkpoint_dir, 'weight_map_V.npy'))

    # load weight maps
    #weight_map_MSGMS = np.load(os.path.join(args.checkpoint_dir, 'weight_map_MSGMS.npy'))
    #weight_map_H = np.load(args.checkpoint_dir + 'weight_map_H.npy')
    #weight_map_S = np.load(args.checkpoint_dir + 'weight_map_S.npy')
    #weight_map_V = np.load(args.checkpoint_dir + 'weight_map_V.npy')
    #weight_maps = [weight_map_H, weight_map_S, weight_map_V]
    binary_weight = np.load(args.checkpoint_dir + 'weight_map_H.npy')

    save_dir = os.path.join(args.save_dir, args.data_type, f'seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    """
    scores = []
    for i in range(len(scores_HSV)):
        combined = []
        for chan in range(3):
            combined.append(scores_HSV[i][chan] * binary_weight) 

        f.plot_HSV_scores(combined, save_dir, i)
        combined = np.asarray(combined)
        combined = np.sum(combined, axis=0)
        #combined = f.normalize_scores(combined)
        scores.append(combined)

    scores = f.normalize_scores(scores)
    #scores = np.asarray(scores)
    """
    scores = scores_MSGMS
    

    threshold_weight = 1.0        # 0.84
    threshold = f.compute_roc_curve(scores, gt_list, gt_mask_list, save_dir, threshold_weight)
    print(f'Threshold: {threshold}')

    f.plot_fig(args, test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir)
    


    #save_dir_recon = './examining/reconstructions/' + args.data_type + '/'
    #os.makedirs(save_dir_recon, exist_ok=True)
    #save_reconstructions(test_imgs, recon_imgs, save_dir_recon)


def test(args, model, test_loader):
    model.eval()
    msgms_score = MSGMS_Score()
    
    scores_MSGMS, test_imgs, gt_list, gt_mask_list, recon_imgs = [], [], [], [], []

    for (data, label, mask) in tqdm(test_loader):

        score_msgms = 0
        with torch.no_grad():
            data = data.to(device)
            for k in args.k_value:
                img_size = data.size(-1)
                N = img_size // k
                Ms_generator = gen_mask([k], 3, img_size)
                Ms = next(Ms_generator)
                inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
                outputs = [model(x) for x in inputs]
                output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
                score_msgms += msgms_score(data, output) / (N**2)

        score_msgms = score_msgms.squeeze().cpu().numpy()

        for i in range(score_msgms.shape[0]):
            score_msgms[i] = gaussian_filter(score_msgms[i], sigma=2)

        scores_MSGMS.extend(score_msgms)

        recon_imgs.extend(output.cpu().numpy())
        test_imgs.extend(data.cpu().numpy())

        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())

    for i in range(len(test_imgs)):
        test_imgs[i] = f.denormalization(test_imgs[i])
        recon_imgs[i] = f.denormalization(recon_imgs[i])

    scores_HSV = compute_HSV_Score(test_imgs, recon_imgs)
    return scores_MSGMS, scores_HSV, test_imgs, recon_imgs, gt_list, gt_mask_list



def compute_HSV_Score(test_imgs, recon_imgs):
    hsv_score = HSV_Score()
    scores = []
    for i in range(len(test_imgs)):
        scores.append(hsv_score(test_imgs[i], recon_imgs[i]))
    return scores



if __name__ == '__main__':
    main()
