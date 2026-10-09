#evaluate file
#joshua wilson
#10-9-26
#summary: loads the saved weights and the test data, computes accuracy and pixel accuracy, and makes the plots. It never trains, which is why you can rerun it in seconds.
#Classification Evaluation: Computes test set accuracy for the CNN and ResNet-18 alongside the majority-class baseline (always-Dog accuracy).
#Segmentation Evaluation: Computes total U-Net pixel accuracy and breaks down accuracy across the three individual mask classes (Pet, Background, Boundary).
#Loss Plotting: Plots and saves loss curves for classification (figures/loss_classifiers.png) and segmentation (figures/loss_unet.png).
#Segmentation Grid: Runs inference on the first three test images and saves a 3×3 visualization grid comparing original images, ground-truth masks, and U-Net predictions (figures/segmentation_results.png).
#Exports summary metrics to results.json.



# GenAI acknowledgment: Used Claude (Anthropic, Claude Sonnet 5.5) to help draft this
# evaluation script (accuracy, baseline, pixel accuracy, plots). I reviewed, ran, and
# tested it myself. Additionally used gemini for debugging.

import json #used for json files
import torch #keep this in if you want to use torch
import matplotlib.pyplot as plt # import for  plotting loss 
from sklearn.metrics import accuracy_score #imports from other file
from data import get_loaders, MASK_NAMES #imports from other files
from models import SimpleCNN, build_resnet18, SmallUNet #imports models

# Use the GPU if PyTorch can see one, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only the test loader is needed here. It is not shuffled, so the image order is the
# same every run (this matters for the "first 3 test images" figure at the end).
_, test_loader = get_loaders(batch_size=16)


def load(model, name):
    """Load saved weights (<name>.pt from train.py) into a freshly built model."""
    model.load_state_dict(torch.load(f"{name}.pt", map_location=device))
    # eval() switches off training-only behavior; this matters for ResNet's BatchNorm,
    # which must use its stored running statistics instead of per-batch statistics.
    return model.to(device).eval()


# ---------- Part 3.1: classifiers (CNN and ResNet-18) ----------
def eval_classifier(model):
    """Return (test accuracy, list of true labels) for a cat/dog classifier."""
    preds, trues = [], []
    with torch.no_grad():                                    # no gradients needed for evaluation
        for images, (labels, masks) in test_loader:          # masks are unused for classification
            outputs = model(images.to(device))               # [B, 2] raw scores
            preds.extend(outputs.argmax(dim=1).cpu().tolist())   # highest score = predicted class
            trues.extend(labels.cpu().tolist())              # true labels, collected in the same order
    return accuracy_score(trues, preds), trues               # fraction of images classified correctly


cnn = load(SimpleCNN(), "cnn") # our cnn
resnet = load(build_resnet18(), "resnet") #our resnet
cnn_acc, true_labels = eval_classifier(cnn)  
res_acc, _ = eval_classifier(resnet)                         # true labels are identical, so discard them
# Majority-class baseline: accuracy of a "model" that always predicts Dog (label 1)
baseline = sum(1 for y in true_labels if y == 1) / len(true_labels)

# Prints a markdown table that can be pasted into the README
print("\n| Model | Test accuracy |\n|---|---|")
print(f"| Regular CNN | {cnn_acc:.4f} |")
print(f"| ResNet-18 | {res_acc:.4f} |")
print(f"| Majority baseline (always Dog) | {baseline:.4f} |")

# ---------- Part 3.2: U-Net pixel accuracy ----------
unet = load(SmallUNet(), "unet")
correct, total = 0, 0            # overall pixel counts
cls_correct = [0, 0, 0]          # per-class counts (extra, used in the README analysis)
cls_total = [0, 0, 0]
with torch.no_grad(): #using torch
    for images, (labels, masks) in test_loader: #loop for labels and mask
        images, masks = images.to(device), masks.to(device)   # masks must be on the same device as predictions
        predicted_masks = unet(images).argmax(dim=1)          # [B,128,128]: best class for every pixel
        correct += (predicted_masks == masks).sum().item()    # pixels predicted correctly
        total += masks.numel()                                # all pixels in this batch
        for c in range(3):                      # 0 = pet, 1 = background, 2 = boundary
            m = masks == c                      # pixels whose TRUE class is c
            cls_total[c] += m.sum().item()
            cls_correct[c] += ((predicted_masks == c) & m).sum().item()   # true class c AND predicted c
pixel_acc = correct / total #calculate pixel accuracy
print(f"\nU-Net pixel accuracy: {pixel_acc:.4f}")
# Per-class accuracy shows what the overall number hides (e.g. weak boundary performance)
for c in range(3):
    print(f"  {MASK_NAMES[c]:>10}: {cls_correct[c] / cls_total[c]:.4f} of true pixels correct "
          f"({cls_total[c] / total:.1%} of all pixels)")

# Save the headline numbers so they can be copied into the README later
with open("results.json", "w") as f: #using json file
    json.dump({"cnn_acc": cnn_acc, "resnet_acc": res_acc, "baseline": baseline,
               "unet_pixel_acc": pixel_acc}, f, indent=2) #

# ---------- Part 4.2: loss plots (read from the files saved by train.py) ----------
loss = {n: json.load(open(f"losses_{n}.json")) for n in ["cnn", "resnet", "unet"]}

# CNN and ResNet share one plot because they solve the same task with the same loss
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(loss["cnn"]) + 1), loss["cnn"], label="Regular CNN")
plt.plot(range(1, len(loss["resnet"]) + 1), loss["resnet"], label="ResNet-18")
plt.xlabel("Epoch"); plt.ylabel("Average training loss"); plt.title("Classifier training loss")
plt.legend(); plt.tight_layout(); plt.savefig("figures/loss_classifiers.png", dpi=150); plt.close()

# U-Net gets its own plot: segmentation loss measures a different task, so its values
# should not be compared with the classifier losses
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(loss["unet"]) + 1), loss["unet"], label="U-Net", color="tab:green")
plt.xlabel("Epoch"); plt.ylabel("Average training loss"); plt.title("U-Net training loss")
plt.legend(); plt.tight_layout(); plt.savefig("figures/loss_unet.png", dpi=150); plt.close()

# ---------- Part 4.3: segmentation figure (first 3 test images) ----------
images, (labels, masks) = next(iter(test_loader))      # first batch; test order is fixed
with torch.no_grad():
    pred = unet(images[:3].to(device)).argmax(dim=1)   # predicted masks for the first 3 images

# Columns: original image, ground-truth mask, predicted mask
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())   # [3,H,W] -> [H,W,3] for imshow
    # Same colormap and vmin/vmax for both mask columns, so a color means the same class in both
    axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
    axes[i, 2].imshow(pred[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
    for j in range(3):
        axes[i, j].axis("off")                         # hide pixel-coordinate axes
axes[0, 0].set_title("Original"); axes[0, 1].set_title("Ground truth"); axes[0, 2].set_title("Predicted")
plt.tight_layout(); plt.savefig("figures/segmentation_results.png", dpi=150); plt.close()
print("\nSaved figures/loss_classifiers.png, loss_unet.png, segmentation_results.png")