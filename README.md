# ViT-implementation
This repository is an attempt to implement ViT from scratch. The goal is to understand the inner workings of BERT by building it step-by-step, rather than relying on pre-built libraries. Through this process, we aim to gain a deeper understanding of its architecture and functionality.

# ViT
The Transformer architecture has become the de facto standard for performing NLP tasks. There have also been attempts to combine it with CNN architectures in the vision domain. The authors of the ViT paper argue that it is possible to achieve good performance by applying it directly to sequences of image patches without relying on CNNs. They state that when pre-trained on large datasets and transferred to specific image recognition benchmark datasets, the **Vision Transformer (ViT)** not only delivers excellent results but also requires significantly fewer resources.

# Model Architecture & Training
Lets briefly discuss about the Model architecture of ViT and how do we train it.
![image](https://github.com/justinshin0204/ViT-impl/assets/93083019/b0696ed7-e59f-4626-84de-b7fca825d435)

I brought the image from the paper. You can see that the image has been divided into 9 patches.
We flatten each 2D patches into 1D vectors, which are then projected into a higher-dimensional space using a learned linear transformation. Additionally, 1D position embeddings are added to these vectors.


These sequences of vectors are then fed into the standard Transformer encoder, as illustrated on the right side of the picture.

Similar to BERT, ViT also adds a learnable embedding at the beginning. The embedding state z<sub>0</sub><sup>L</sup> is used as the image representation y. During both pre-training and fine-tuning, a classification head is attached to  z<sub>0</sub><sup>L</sup>. For pre-training, a single-layer MLP is used, while a simple linear layer is employed during the fine-tuning stage.

### Issues with CNNs

CNNs guide us on how to view images by initially focusing on local regions and gradually expanding to extract features. However, there's no guarantee that this is the best approach.

### Vision Transformer (ViT) Approach

ViT, on the other hand, lets the AI determine when and where to look, essentially saying, "You figure it out!" This black-box approach tends to perform better with larger datasets, as fewer inductive biases can be beneficial with more data.

Now we know how it works, let implement this.



# Implementation

## Packages
```py
from google.colab import drive
drive.mount('/content/drive')
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from einops import rearrange
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
```
import the packages needed
```py
BATCH_SIZE = 4096
LAMBDA = 1e-1
EPOCH = 50
scheduler_name = 'Cos'
LR = 1e-4
criterion = nn.CrossEntropyLoss()
save_model_path = '/content/drive/MyDrive/Colab Notebooks/results/ViT_CIFAR10.pt'
save_history_path = '/content/drive/MyDrive/Colab Notebooks/results/ViT_CIFAR10_history.pt'
```
set the hyperparameters and scheduler

## Data augmentation / DataLoader 
```py
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

transform_test = transforms.ToTensor()

train_DS = datasets.CIFAR10(root = '/content/drive/MyDrive/Colab Notebooks/data', train=True, download=True, transform=transform_train)
train_DS, val_DS = torch.utils.data.random_split(train_DS, [45000, 5000])
test_DS = datasets.CIFAR10(root = '/content/drive/MyDrive/Colab Notebooks/data', train=False, download=True, transform=transform_test)

train_DL = torch.utils.data.DataLoader(train_DS, batch_size = BATCH_SIZE, shuffle = True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size = BATCH_SIZE, shuffle = True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size = BATCH_SIZE, shuffle = True)
```

Feel free to customize the transformation pipeline as per your requirements. <br>
This augmentation setup aims to provide diverse and robust training samples to improve the generalization ability of the model.

# Model Architecture
## Multi-Head attention
```py
class MHA(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.sqrt(torch.tensor(hidden_dim / n_heads))

        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

        if self.fc_q.bias is not None:
            nn.init.constant_(self.fc_q.bias, 0)
        if self.fc_k.bias is not None:
            nn.init.constant_(self.fc_k.bias, 0)
        if self.fc_v.bias is not None:
            nn.init.constant_(self.fc_v.bias, 0)
        if self.fc_o.bias is not None:
            nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, x):

        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = rearrange(Q, 'num word (head dim) -> num head word dim', head=self.n_heads)
        K = rearrange(K, 'num word (head dim) -> num head word dim', head=self.n_heads)
        V = rearrange(V, 'num word (head dim) -> num head word dim', head=self.n_heads)

        attention_score = Q @ K.transpose(-2, -1) / self.scale

        attention_weights = torch.softmax(attention_score, dim=-1)

        attention = attention_weights @ V

        x = rearrange(attention, 'num head word dim -> num word (head dim)')
        x = self.fc_o(x)

        return x, attention_weights
```
## Feed forward
```py
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(hidden_dim, d_ff),
                                    nn.GELU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, hidden_dim))

    def forward(self, x):
        x = self.linear(x)
        return x
```
Notice that we still refer to the number of consecutive sequences as "word".
These are just standard Multi-Head Attention (MHA) and Feed Forward (FF) networks, initialized using Xavier initialization—nothing unusual.
## Transformer Encoder
```py
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten_LN = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_atten = MHA(hidden_dim, n_heads)

        self.FF_LN = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.FF = FeedForward(hidden_dim, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):

        residual = self.self_atten_LN(x)
        residual, _ = self.self_atten(residual)
        residual = self.dropout(residual)
        x = x + residual

        residual = self.FF_LN(x)
        residual = self.FF(residual)
        residual = self.dropout(residual)
        x = x + residual

        return x

class Encoder(nn.Module):
    def __init__(self, seq_length, n_layers, hidden_dim, d_ff, n_heads, drop_p):
        super().__init__()

        self.pos_embedding = nn.Parameter(0.02 * torch.randn(seq_length, hidden_dim))
        self.dropout = nn.Dropout(drop_p)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, d_ff, n_heads, drop_p) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, src, atten_map_save=False):

        x = src + self.pos_embedding.expand_as(src)
        x = self.dropout(x)

        for layer in self.layers:
            x= layer(x)
        x = x[:, 0, :]  # num x word xdim
        x = self.ln(x)

        return x
```

## ViT
```py
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, hidden_dim, d_ff, n_heads, representation_size = None, drop_p = 0., num_classes = 1000):
        super().__init__()

        self.hidden_dim = hidden_dim # vector dimension 

        seq_length = (image_size // patch_size) ** 2 + 1 # +1 for cls token
        self.class_token = nn.Parameter(torch.zeros(hidden_dim)) # cls token
        self.input_embedding = nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size) ## Use nn.Conv2d to do the embedding at once
        self.encoder = Encoder(seq_length, n_layers, hidden_dim, d_ff, n_heads, drop_p)

        heads_layers = []
        if representation_size is None: 
            self.head = nn.Linear(hidden_dim, num_classes) # fine-tune 
        else:
            # pre-train
            self.head = nn.Sequential(nn.Linear(hidden_dim, representation_size), 
                                      nn.Tanh(), 
                                      nn.Linear(representation_size, num_classes))

        # fan_in=3 x kernel size 
        fan_in = self.input_embedding.in_channels * self.input_embedding.kernel_size[0] * self.input_embedding.kernel_size[1]
        # weight init
        nn.init.trunc_normal_(self.input_embedding.weight, std=math.sqrt(1 / fan_in))
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)
      
        if representation_size is None:
            nn.init.zeros_(self.head.weight) # fine-tune => zero-init
            nn.init.zeros_(self.head.bias) 
        else: # pre-training 
            fan_in = self.head[0].in_features
            nn.init.trunc_normal_(self.head[0].weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.head[0].bias)

    def forward(self, x): 

        x = self.input_embedding(x) 
        x = rearrange(x, 'num dim h w -> num (h w) dim') 

        batch_class_token = self.class_token.expand(x.shape[0], 1, -1)
        x = torch.cat([batch_class_token, x], dim=1) 

        enc_out = self.encoder(x) # 

        x = self.head(enc_out)
```
In this implementation,images are embedded using a convolutional layer, and a class token is added. The sequence is processed by an encoder with multiple layers. For output, a linear layer is used for fine-tuning or an MLP with Tanh for pre-training. Weights are initialized with trunc-normal and biases with zeros for stability. This setup leverages Transformer architecture for effective image classification. <br>
the input => number of batch x (( image_size / patch_size)**2 +1) x hidden_dim <br>
the output => number of classes

## Training
```py
def Train(model, train_DL, val_DL, criterion, optimizer, scheduler = None):
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}
    best_loss = 9999
    for ep in range(EPOCH):
        model.train() # train mode
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer = optimizer, scheduler = scheduler)
        loss_history["train"] += [train_loss]
        acc_history["train"] += [train_acc]

        model.eval() # test mode
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion)
            loss_history["val"] += [val_loss]
            acc_history["val"] += [val_acc]
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model,
                            "ep": ep,
                            "optimizer": optimizer,
                            "scheduler": scheduler}, save_model_path)
        # print loss
        print(f"Epoch: {ep+1}, current_LR = {optimizer.param_groups[0]['lr']:.8f}")
        print(f"train loss: {train_loss:.5f}, "
              f"val loss: {val_loss:.5f} \n"
              f"train acc: {train_acc:.1f} %, "
              f"val acc: {val_acc:.1f} %")
        print("-" * 20)

    torch.save({"loss_history": loss_history,
                "acc_history": acc_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE}, save_history_path)

def Test(model, test_DL, criterion):
    model.eval() # test mode
    with torch.no_grad():
        test_loss, test_acc, rcorrect = loss_epoch(model, test_DL, criterion)
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_acc,1)} %)")
    return round(test_acc,1)

def loss_epoch(model, DL, criterion, optimizer = None, scheduler = None):
    N = len(DL.dataset) # the number of data
    rloss=0; rcorrect = 0
    for x_batch, y_batch in tqdm(DL, leave=False):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # inference
        y_hat = model(x_batch)
        # loss
        loss = criterion(y_hat, y_batch)
        # update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # loss accumulation
        loss_b = loss.item() * x_batch.shape[0]
        rloss += loss_b
        # accuracy accumulation
        pred = y_hat.argmax(dim=1)
        corrects_b = torch.sum(pred == y_batch).item()
        rcorrect += corrects_b
    loss_e = rloss/N
    accuracy_e = rcorrect/N * 100

    return loss_e, accuracy_e, rcorrect
```
Now, lets train our model.
```py
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=LR_init, weight_decay=LAMBDA)
scheduler = CosineAnnealingLR(optimizer, T_max = int(len(train_DS)*EPOCH/BATCH_SIZE))
Train(model, train_DL, val_DL, criterion, optimizer, scheduler)
```
