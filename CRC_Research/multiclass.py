#1 Visualization
import os #1.1
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd #1.2
import seaborn as sns

import time

#2 - Creation
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import itertools


#ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#extra
from tqdm import tqdm

#route to the images
base_dir = "C:\\Users\\riley\\OneDrive\\Desktop\\CRC_Research\\Kather_texture_2016_image_tiles_5000"
#'/Users/rileyroggenkamp/Documents/GitHub/Colorectal_Histology_MNIST/Kather_texture_2016_image_tiles_5000';

#checks amount of datatypes and sorts them
img_labels = [i for i in os.listdir(base_dir) if not i.startswith('.')]
print('There are {} classes in this dataset:\n{}'.format(len(img_labels), img_labels))

#array (?) of each file type // os.path.join finds the path, glob() searches the directory to find the array
tumor_files = glob(os.path.join(base_dir, img_labels[0], '*.tif'))
stroma_files = glob(os.path.join(base_dir, img_labels[1], '*.tif'))
complex_files = glob(os.path.join(base_dir, img_labels[2], '*.tif'))
lympho_files = glob(os.path.join(base_dir, img_labels[3], '*.tif'))
debris_files = glob(os.path.join(base_dir, img_labels[4], '*.tif'))
mucosa_files = glob(os.path.join(base_dir, img_labels[5], '*.tif'))
adipose_files = glob(os.path.join(base_dir, img_labels[6], '*.tif'))
empty_files = glob(os.path.join(base_dir, img_labels[7], '*.tif'))
img_files = [tumor_files, stroma_files, complex_files, lympho_files, debris_files, mucosa_files, adipose_files, empty_files]
total_files = [img for folder in img_files for img in folder]


print('Total number of images in this dataset: {}'.format(len(total_files)))
print ('-'*50)
for i in np.arange(8):
    print('Number of images for {} category: {}'.format(img_labels[i].split('_')[1], len(img_files[i])))

def readImage_rgb(img_path):
    '''OpenCV loads color images in BGR mode and converts to RGB mode for visualization;
       output: (H x W x n_channel)'''
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
    
img = readImage_rgb(img_files[0][0])
print(img.shape)

#summarizes images 
def image_summary(image_paths, img_label):
    img_dict = {}
    print(len(image_paths))
    for i in range(len(image_paths)):
        print(i)
        img_path = image_paths[i]
        img_dict[img_path] = {}
        img = cv2.imread(img_path)
        img_dict[img_path]['label'] = img_label
        print(img_label)
        img_dict[img_path]['max'] = img.max()
        #print("max" + str(img.max()))
        img_dict[img_path]['min'] = img.min()
        #print("min" + str(img.min()))
        
        channel_mean = img.mean(axis = (0,1), keepdims = True).squeeze()
        channel_std = img.std(axis = (0,1), keepdims = True).squeeze()
        channel_q01 = np.quantile(img, 0.1, axis=(0,1), keepdims=True).squeeze()
        channel_q09 = np.quantile(img, 0.9, axis=(0,1), keepdims=True).squeeze()
        img_dict[img_path]['red_mean'], img_dict[img_path]['green_mean'], img_dict[img_path]['blue_mean'] = channel_mean
        img_dict[img_path]['red_std'], img_dict[img_path]['green_std'], img_dict[img_path]['blue_std'] = channel_std
        img_dict[img_path]['red_q01'], img_dict[img_path]['green_q01'], img_dict[img_path]['blue_q01'] = channel_q01
        img_dict[img_path]['red_q09'], img_dict[img_path]['green_q09'], img_dict[img_path]['blue_q09'] = channel_q09

    print('one done')
    img_df = pd.DataFrame.from_dict(img_dict, orient = 'index')
    return img_df

""""
tumor_files_df = image_summary(tumor_files, 'TUMOR')
tumor_files_df.to_csv("tumor_files_df.csv")
stroma_files_df = image_summary(stroma_files, 'STROMA')
stroma_files_df.to_csv("stroma_files_df.csv")
complex_files_df = image_summary(complex_files, 'COMPLEX')
complex_files_df.to_csv("complex_files_df.csv")
lympho_files_df = image_summary(lympho_files, 'LYMPHO')
lympho_files_df.to_csv("lympho_files_df.csv")
debris_files_df = image_summary(debris_files, 'DEBRIS')
debris_files_df.to_csv("debris_files_df.csv")
mucosa_files_df = image_summary(mucosa_files, 'MUCOSA')
mucosa_files_df.to_csv("mucosa_files_df.csv")
adipose_files_df = image_summary(adipose_files, 'ADIPOSE')
adipose_files_df.to_csv("adipose_files_df.csv")
empty_files_df = image_summary(empty_files, 'EMPTY')
empty_files_df.to_csv("empty_files_df.csv")
"""

tumor_files_df = pd.read_csv("tumor_files_df.csv", usecols=[0, 1])
stroma_files_df = pd.read_csv("stroma_files_df.csv", usecols=[0, 1])
complex_files_df = pd.read_csv("complex_files_df.csv", usecols=[0, 1])
lympho_files_df = pd.read_csv("lympho_files_df.csv", usecols=[0, 1])
debris_files_df = pd.read_csv("debris_files_df.csv", usecols=[0, 1])
mucosa_files_df = pd.read_csv("mucosa_files_df.csv", usecols=[0, 1])
adipose_files_df = pd.read_csv("adipose_files_df.csv", usecols=[0, 1])
empty_files_df = pd.read_csv("empty_files_df.csv", usecols=[0, 1])

pixel_df = pd.concat([tumor_files_df, stroma_files_df, complex_files_df, lympho_files_df, debris_files_df, mucosa_files_df, adipose_files_df, empty_files_df])
print("Pixel Analysis:")
print(pixel_df.shape)

train_val_test_df = pixel_df.copy()
total_num = train_val_test_df.shape[0]
split1 = int(total_num*0.7) ##HERE #.6 old: .7
split2 = int(total_num*0.85) #.8 old: .85
print(split2)
indices = np.arange(total_num)

np.random.seed(123)
np.random.shuffle(indices)
train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)


train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(144),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
BATCH_SIZE = 32
train_path = base_dir
train_data = datasets.ImageFolder(train_path, transform = train_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler = val_sampler)
test_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, sampler = test_sampler)

print('Number of images in training set: {}'.format(len(train_loader.sampler)))
print('Number of images in validation set: {}'.format(len(val_loader.sampler)))
print('Number of images in testing set: {}'.format(len(test_loader.sampler)))

#2 build CNN from scratch
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1_1 = nn.Conv2d(3,32,3,padding=1)
        self.conv1_2 = nn.Conv2d(32,32,3,padding=1)
        self.conv1_3 = nn.Conv2d(32,32,3,padding=1)

        self.conv2_1 = nn.Conv2d(32,64,3,padding=1)
        self.conv2_2 = nn.Conv2d(64,64,3,padding=1)
        self.conv2_3 = nn.Conv2d(64,64,3,padding=1)

        self.conv3_1 = nn.Conv2d(64,128,3,padding=1)
        self.conv3_2 = nn.Conv2d(128,128,3,padding=1)
        self.conv3_3 = nn.Conv2d(128,128,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*18*18,256)
        self.fc2 = nn.Linear(256,8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1,128*18*18)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device.type))
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should be >= 1
print(torch.cuda.get_device_name(0))  # Your GPU name
    
model = Model()
# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path):
    '''the train function will perform both forward and backpropagation on training and validation datasets.
    Output: trained model with the lowest val_loss and dataframe containing train_loss/train_acc and val_loss/val_acc for each epoch'''
    epoch_dict={}
    valid_loss_min = np.inf
    model = model.to(device)
    for epoch in np.arange(n_epochs):
        # Initiate loss and accuracy values
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        epoch_dict[epoch] = {}
        # Training
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            ## Update loss and acc values for the training process
            train_loss += loss.item()*images.shape[0]
            _, top_class = output.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor)).item()

        # Validation
        else:
            # turn off gradients
            with torch.no_grad():
                # set model to evaluation mode
                model.eval()
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    loss = criterion(output, labels)
                    ## Update loss and acc values for the validation process
                    val_loss += loss.item()*images.shape[0]
                    _, top_class = output.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_acc += torch.mean(equals.type(torch.FloatTensor)).item()

                ### write training/validation loss/accuracy
                epoch_dict[epoch]['Train_Loss'] = train_loss/len(train_loader.sampler)
                epoch_dict[epoch]['Train_Accuracy'] = train_acc/len(train_loader)
                epoch_dict[epoch]['Val_Loss'] = val_loss/len(val_loader.sampler)
                epoch_dict[epoch]['Val_Accuracy'] = val_acc/len(val_loader)

                print("Epoch: {}/{}.. ".format(epoch+1, n_epochs),
                      "Train Loss: {:.3f}.. ".format(train_loss/len(train_loader.sampler)),
                      "Train Accuracy: {:.3f}.. ".format(train_acc/len(train_loader)),
                      "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader.sampler)),
                      "Validation Accuracy: {:.3f}".format(val_acc/len(val_loader)))
                
                ## save the model with the lowest val_loss and update valid_loss_min
                if val_loss <= valid_loss_min:
                    print('Validation loss decreased -- Saving model -- \n')
                    torch.save(model.state_dict(), save_path)
                    valid_loss_min = val_loss

    epoch_df = pd.DataFrame.from_dict(epoch_dict, orient = 'index')
    epoch_df['Epoch'] = np.arange(n_epochs)+1
    epoch_df.to_csv(csv_path)  # Save as CSV (new, here!)
    return model, epoch_df


n_epochs = 50
save_path = 'models/4multiclass.pth'
csv_path = 'models/4multiclass_stats.csv'
model, epoch_df = train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path)
print("Done training")

#2.4 Visualize loss & accuracy curves
epoch_df = pd.read_csv(csv_path)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epoch_df.Epoch, epoch_df.Train_Loss, label = 'Training loss')
plt.plot(epoch_df.Epoch, epoch_df.Val_Loss, label = 'Validation loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.title('Training and Validation Loss', fontsize=16)

plt.subplot(1,2,2)
# Find the highest validation accuracy and its corresponding epoch
max_val_accuracy = epoch_df.Val_Accuracy.max()
max_val_accuracy_epoch = epoch_df.Epoch[epoch_df.Val_Accuracy.idxmax()]

# Add horizontal line at the highest validation accuracy
plt.axhline(y=max_val_accuracy, color='r', linestyle='--', label=f'Highest Val Accuracy: {max_val_accuracy:.2f}')

# Optionally, add text to display the highest validation accuracy value
plt.text(max_val_accuracy_epoch, max_val_accuracy, f'{max_val_accuracy:.2f}', color='r', fontsize=12, verticalalignment='bottom')

plt.plot(epoch_df.Epoch, epoch_df.Train_Accuracy, label = 'Training Accuracy')
plt.plot(epoch_df.Epoch, epoch_df.Val_Accuracy, label = 'Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.title('Training and Validation Accuracy', fontsize=16)
plt.show()

model = Model()
state_dict = torch.load(save_path)

model.load_state_dict(state_dict)
for param in model.parameters():
    param.requires_grad=False

#2.6 Load Images
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
## No augmentation
test_transforms = transforms.Compose([
    transforms.Resize((144,144)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
test_data = ImageFolderWithPaths(train_path, transform = test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, sampler = test_sampler)
print('Number of images in testing set: {}'.format(len(test_loader.sampler)))


#2.7 Make Prediction
def model_eval(test_loader, model, criterion, device, classes):
    '''Apply the trained model to testing dataset and summarize the overall test_loss/test_acc
    also output testing accuracy for each category and the prediction result for each sample as well as the probability'''
    # Initiate loss and accuracy values
    test_loss = 0.0
    test_acc = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
  
    dt = np.dtype(int)
    true_label = np.array([], dtype = dt)
    pred_label = np.array([], dtype = dt)
    path_array = np.array([])
    prob_array = np.array([])
    probs_array = np.array(classes).reshape(1,-1)
    
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for images, labels, paths in test_loader:
            path_array = np.append(path_array, np.array(paths))
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()*images.shape[0]
        
            softmax = nn.Softmax(dim=1)
            output_softmax = softmax(output)
            top_probs, preds = output_softmax.topk(1,dim=1)
            equals = preds == labels.view(*preds.shape)
            correct = np.squeeze(equals)
            test_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            for i in range(len(images)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
            if device.type == 'cpu':
                true_label = np.append(true_label, labels)
                pred_label = np.append(pred_label, preds)
                prob_array = np.append(prob_array, top_probs)
                probs_array = np.concatenate((probs_array, output_softmax))
      
            else:
                true_label = np.append(true_label, labels.cpu())
                pred_label = np.append(pred_label, preds.cpu())
                prob_array = np.append(prob_array, top_probs.cpu())
                probs_array = np.concatenate((probs_array, output_softmax.cpu()))
            
      
    test_loss = test_loss/len(test_loader.sampler)
    test_acc = test_acc/len(test_loader)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('Test Accuracy (Overall): {:.3f}%  ({}/{})\n'.format(np.sum(class_correct)/np.sum(class_total)*100, np.sum(class_correct), np.sum(class_total)))
  
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of {}: {:.3f}% ({}/{})'.format(classes[i], class_correct[i]/class_total[i]*100,class_correct[i], class_total[i]))
        else:
            print('Test Accuracy of {}: N/A'.format(classes[i]))
    summary_df = pd.DataFrame({'category': classes, 'correct': class_correct, 'total': class_total})
    pred_df = pd.DataFrame({'file_path': path_array, 'true_label': true_label, 'prediction': pred_label, 'prob': prob_array})
    probs_df = pd.DataFrame(probs_array[1:], columns = probs_array[0])
    probs_df['file_path'] = path_array
    pred_df = pred_df.merge(probs_df, left_on = 'file_path', right_on = 'file_path')
    return summary_df, pred_df
criterion = nn.CrossEntropyLoss()
classes = test_data.classes
summary_df, pred_df = model_eval(test_loader, model, criterion, device, classes)


summary_df['accuracy'] = np.round(summary_df.correct/summary_df.total*100,2)
fig = plt.figure(figsize = (20,8))
ax = fig.add_subplot(1,1,1)
width = 0.75 # the width of the bars 
ind = np.arange(len(classes))  # the x locations for the groups
ax.bar(ind, summary_df.accuracy, width, edgecolor='black', color = sns.color_palette('hls', 8))
plt.xticks(ind, labels = classes, fontsize=18, fontweight='bold', rotation=45)
plt.yticks(fontsize=16)
plt.xlabel('Category', fontsize=18)
plt.ylabel('Accuracy%', fontsize=18)
for i, v in enumerate(summary_df.accuracy):
    ax.text(i-0.25, v + 1, str(np.round(v,2))+'%', fontweight='bold', fontsize=18)
ax.plot([-0.5, 7.4], [86.174, 86.174], "k--", linewidth=2)
ax.text(7.4, 83, 'Avg:\n'+str(86.174)+'%', fontweight='bold', fontsize=18)
plt.show()


cm = confusion_matrix(pred_df.true_label, pred_df.prediction)
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):    
    plt.figure(figsize=(10,7))  # Smaller figure size
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)  # Reduce font size of title
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, rotation = 45, fontweight='bold')  # Smaller font size for ticks
    plt.yticks(tick_marks, classes, fontsize=10, fontweight='bold')  # Smaller font size for ticks

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:d} \n({:.2f}%)'.format(cm[i, j], cm[i,j]/cm.sum()*100), 
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=10)  # Smaller font for text in cells

    plt.ylabel('True label', fontsize=12)  # Smaller font for label
    plt.xlabel('Predicted label', fontsize=12)  # Smaller font for label
    
    # Adjust layout with more padding
    plt.tight_layout(pad=4.0)  # Reduce padding to make everything a bit tighter
    plt.subplots_adjust(right=0.85, top=0.88)  # Keep margins adjusted for smaller size

    plt.show()
plot_confusion_matrix(cm, classes, title = 'Confusion Matrix')



#ROC CURVE
classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']  # Add all class names here

# Assuming `true_labels` is a multi-class array (you can convert the `true_label` column to a numpy array)
true_labels = pred_df['true_label'].values  # True labels should be in 'true_label' col
print(pred_df.columns)
fpr = {}
tpr = {}
roc_auc = {}

# Color palette for plotting
palette = sns.color_palette('hls', len(classes))

# Loop over each class to compute ROC curve and AUC
for i, class_name in enumerate(classes):
    # Check if the class column exists in pred_df
    if class_name in pred_df.columns:
        probs = pred_df[class_name]  # predicted probabilities for class `i`

        # Ensure the probabilities are numeric (in case they were loaded as strings)
        if probs.dtype != float:
            pred_df[class_name] = pred_df[class_name].astype(float)
            probs = pred_df[class_name]

        # Compute ROC curve
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, probs)

        # Compute AUC
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for the current class
        plt.plot(fpr[i], tpr[i], color=palette[i], lw=2, label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    else:
        print(f"Error: Column '{class_name}' not found in pred_df")

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Multiclass ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)

# Show the plot
plt.show()

# Optional: Display AUC values for each class
print("\nAUC values per class:")
for class_name, auc_value in zip(classes, roc_auc.values()):
    print(f"{class_name}: {auc_value:.2f}")
