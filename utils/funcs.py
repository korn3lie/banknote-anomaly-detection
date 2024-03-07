import numpy as np
import torch, os, cv2
from utils.utils import print_log
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from skimage.segmentation import mark_boundaries
from skimage import morphology, measure
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# ---------------------------------------------------------------------------------------------------------------

def precision_recall_plot(precision,recall,save_dir):
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'precision_recall.png'), dpi=100)

def plot_confusion_matrix(y_true, y_pred, threshold, save_dir):

    y_pred = np.where(np.array(y_pred) > threshold, 1, 0).tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fig, ax = plt.subplots()

    # Create the table
    table_data = np.array([[tn, fp], [fn, tp]])
    table = ax.table(cellText=table_data,
                 rowLabels=["Actual Negative", "Actual Anomaly"],
                 colLabels=["Predicted Negative", "Predicted Anomaly"],
                 loc="center")
    ax.axis("off")
    fig.suptitle("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches="tight", pad_inches=0.1)


def compute_roc_curve(scores, gt_list, gt_mask_list, save_dir, threshold_weight=1.0):

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_rocauc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_rocauc))
    plt.plot(fpr, tpr, label=f'image_ROC AUC: {img_rocauc:3f}')
    plt.legend(loc="lower right")

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    threshold *= threshold_weight

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (pixel_rocauc))

    plt.plot(fpr, tpr, label=f'pixel_ROC AUC: {pixel_rocauc:3f}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)

    plot_confusion_matrix(gt_list, img_scores, threshold, save_dir)
    #precision_recall_plot(precision,recall,save_dir)

    return threshold

# ---------------------------------------------------------------------------------------------------------------

def normalize_scores(scores):
    scores = np.asarray(scores)
    min_score, max_score = scores.min(), scores.max()
    scores = (scores - min_score) / (max_score - min_score)
    return scores

# ---------------------------------------------------------------------------------------------------------------

def calculate_score_weight_map(scores, save_path):
    weight_map = np.mean(scores, axis=0)
    weight_map = (1 - weight_map)
    #print(weight_map.shape)
    np.save(save_path, weight_map)

# ---------------------------------------------------------------------------------------------------------------

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def denormalization_torch(x, device):
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    
    mean = mean.to(x.device)
    std = std.to(x.device)
    x = (((x.permute(1, 2, 0) * std) + mean) * 255.0).to(torch.uint8)
    return x

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# ---------------------------------------------------------------------------------------------------------------

class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

# ---------------------------------------------------------------------------------------------------------------
        
def plot_HSV_scores(scores, save_dir, k):
    ch = ['HUE', 'SATURATION', 'VALUE']
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(scores[i])
        plt.title(f'{ch[i]} score\n-------------------\nMax: {scores[i].max():.3f}    ||    Sum: {scores[i].sum():.3f}')
        plt.colorbar()

    plt.savefig(os.path.join(save_dir, f'HSV_weighted_scores_{str(k).zfill(5)}.png'), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------------------------------------------

def plot_fig(args, test_imgs, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255
    vmin = scores.min() * 255

    for i in range(num):
        #img = denormalization(test_imgs[i])
        #recon_img = denormalization(recon_imgs[i])
        img = test_imgs[i]
        recon_img = recon_imgs[i]

        gt = gts[i].transpose(1, 2, 0).squeeze()

        heat_map = scores[i] * 255

        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        #kernel = morphology.disk(2)
        #mask = morphology.opening(mask, kernel)
        mask *= 255

        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(2, 3, figsize=(20, 11))


        for ax_j in ax_img:
            for ax_i in ax_j:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

        # First row -------------------------------------------------------------
        ax_img[0, 0].imshow(img)
        ax_img[0, 0].set_title('Original Image')

        ax_img[0, 1].imshow(recon_img)
        ax_img[0, 1].set_title('Reconstruction')

        ax_img[0, 2].imshow(gt, cmap='gray')
        ax_img[0, 2].set_title('GroundTruth')

        # Second row -------------------------------------------------------------
        ax_img[1, 0].imshow(img, cmap='gray', interpolation='none')
        ax_img[1, 0].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', vmin=vmin, vmax=vmax)
        ax_img[1, 0].set_title(f'Heat Map\nmax global : {vmax:.1f}\nthreshold : {threshold*255:.1f}\nmax local : {heat_map.max():.1f}')

        ax_img[1, 1].imshow(vis_img)
        ax_img[1, 1].set_title('Segmentation result')

        ax_img[1, 2].imshow(mask, cmap='gray')
        ax_img[1, 2].set_title('Predicted mask')

        fig_img.savefig(os.path.join(save_dir, f'{str(i).zfill(5)}.png'), dpi=100)
        plt.close()

# ---------------------------------------------------------------------------------------------------------------
def save_reconstructions(test_imgs, recon_imgs, save_dir):
    
    n = len(test_imgs)
    for i in range(n):

        img = denormalization(test_imgs[i])
        recon_img = denormalization(recon_imgs[i])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        # Save original image
        output_path = os.path.join(save_dir, f'{str(i).zfill(5)}.png')
        cv2.imwrite(output_path, img)

        # Save reconstruction
        output_path = os.path.join(save_dir, f'{str(i).zfill(5)}_recon.png')
        cv2.imwrite(output_path, recon_img)
