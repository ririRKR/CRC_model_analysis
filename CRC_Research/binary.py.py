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

#extra
from tqdm import tqdm

#route to the images
base_dir = "C:\\Users\\riley\\OneDrive\\Desktop\\CRC_Research\\Kather_texture_2016_image_tiles_5000"
#'/Users/rileyroggenkamp/Documents/GitHub/Colorectal_Histology_MNIST/Kather_texture_2016_image_tiles_5000';

#checks amount of datatypes and sorts them
img_labels = [i for i in os.listdir(base_dir) if not i.startswith('.')]
print('There are {} classes in this dataset:\n{}'.format(len(img_labels), img_labels))

cancerous_labels = {'01_TUMOR', '03_COMPLEX', '04_LYMPHO'} #n
non_cancerous_labels = {'02_STROMA', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY'}

#array (?) of each file type // os.path.join finds the path, glob() searches the directory to find the array
tumor_files = glob(os.path.join(base_dir, img_labels[0], '*.tif'))
stroma_files = glob(os.path.join(base_dir, img_labels[1], '*.tif'))
complex_files = glob(os.path.join(base_dir, img_labels[2], '*.tif'))
lympho_files = glob(os.path.join(base_dir, img_labels[3], '*.tif'))
debris_files = glob(os.path.join(base_dir, img_labels[4], '*.tif'))
mucosa_files = glob(os.path.join(base_dir, img_labels[5], '*.tif'))
adipose_files = glob(os.path.join(base_dir, img_labels[6], '*.tif'))
empty_files = glob(os.path.join(base_dir, img_labels[7], '*.tif'))


image_paths = (
    tumor_files +
    stroma_files +
    complex_files +
    lympho_files +
    debris_files +
    mucosa_files +
    adipose_files +
    empty_files
)

def binary_label_from_path(image_path): #n
    class_folder = os.path.basename(os.path.dirname(image_path))
    return 1 if class_folder in cancerous_labels else 0

data = [(path, binary_label_from_path(path)) for path in image_paths] #n

cancerous_files = []
for label in cancerous_labels:
    cancerous_files.extend(list(glob(os.path.join(base_dir, label, '*.tif'))))  # Convert to list

non_cancerous_files = []
for label in non_cancerous_labels:
    non_cancerous_files.extend(list(glob(os.path.join(base_dir, label, '*.tif'))))  # Convert to list

print('Cancerous File List Length: ' + str(len(cancerous_files)))
print('Nonancerous File List Length: ' + str(len(non_cancerous_files)))

image_paths = cancerous_files+non_cancerous_files

#array list of all of the file lists
img_files = [tumor_files, stroma_files, complex_files, lympho_files, debris_files, mucosa_files, adipose_files, empty_files]
#flattens file lists into a single array
total_files = [img for folder in img_files for img in folder]

#prints number of images in dataset
print('Total number of images in this dataset: {}'.format(len(total_files))) #.format places strings into other strings, len(total_files) gets length of a list in a string
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

#only need to run if adding new files
#tumor_files_df = image_summary(tumor_files, 'TUMOR')
#stroma_files_df = image_summary(stroma_files, 'STROMA')
#complex_files_df = image_summary(complex_files, 'COMPLEX')
#lympho_files_df = image_summary(lympho_files, 'LYMPHO')
#debris_files_df = image_summary(debris_files, 'DEBRIS')
#mucosa_files_df = image_summary(mucosa_files, 'MUCOSA')
#adipose_files_df = image_summary(adipose_files, 'ADIPOSE')
#empty_files_df = image_summary(empty_files, 'EMPTY')

#cancerous_files_df = image_summary(cancerous_files, 'CANCEROUS')
#cancerous_files_df.to_csv("cancerous_files_df.csv")
cancerous_files_df = pd.read_csv("cancerous_files_df.csv", usecols=[0, 1])
#non_cancerous_files_df = image_summary(non_cancerous_files, 'NONCANCEROUS')
#non_cancerous_files_df.to_csv("non_cancerous_files_df.csv")
non_cancerous_files_df = pd.read_csv("non_cancerous_files_df.csv", usecols=[0, 1])

#print("DONE WITH CSV FOR C/N")

#pixel_df = pd.concat([tumor_files_df, stroma_files_df, complex_files_df, lympho_files_df, debris_files_df, mucosa_files_df, adipose_files_df, empty_files_df])
pixel_df = pd.concat([cancerous_files_df, non_cancerous_files_df])
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

#random cropping and orientation, normalization
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(144),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5,0.5], [0.5,0.5,0.5])
])
BATCH_SIZE = 32
train_path = base_dir
train_data = datasets.ImageFolder(train_path, transform = train_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler = train_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler = val_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, sampler = test_sampler)

print('Number of images in training set: {}'.format(len(train_loader.sampler)))
print('Number of images in validation set: {}'.format(len(val_loader.sampler)))
print('Number of images in testing set: {}'.format(len(test_loader.sampler)))

#visualization post augmentation
#def imshow_transform(img):
  #  img = img*0.5+0.5
  #  plt.imshow(np.transpose(img, (1,2,0)))

#images, labels = next(iter(train_loader))
#images = images.numpy().squeeze()
classes = train_data.classes
#palette = sns.color_palette('tab10', 8)

#fig = plt.figure(figsize = (25,8))
#for idx in np.arange(20):
#    ax = fig.add_subplot(2,10,idx+1,xticks=[], yticks=[])
   # imshow_transform(images[idx])
 #   ax.set_title(classes[labels.numpy()[idx]], fontsize=18)
#fig.suptitle('H&E images for colorectal histology post augmentation', fontsize = 32)

#2.2 Build CNN from scratch
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
        self.fc2 = nn.Linear(256,1) #2 (?)
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


#2.3 Training
#utilize GPU over CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device.type))

model = Model()
# specify loss function #newest
pos_weight = torch.tensor(3125 / 1875)  # = 5/3 ≈ 1.6667
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
print(pos_weight.shape)
#riterion = nn.BCEWithLogitsLoss() ##nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def relabel_labels(original_label):
    '''Relabel multiclass labels to binary.'''
    if original_label in [0, 2, 3]:
        return 1  # Cancerous (class 1)
    elif original_label in [1, 4, 5, 6, 7]:
        return 0  # Non-cancerous (class 0)
    return original_label

def train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path):
    '''The train function will perform both forward and backpropagation on training and validation datasets.
    Output: trained model with the lowest val_loss and dataframe containing train_loss/train_acc and val_loss/val_acc for each epoch'''
    
    epoch_dict = {}
    valid_loss_min = np.inf
    model = model.to(device)
    
    for index, epoch in enumerate(np.arange(n_epochs)):
        print("Epoch: {}, Value from arange: {}".format(epoch, index))
        
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
            
            # Relabel the original labels into binary labels (0 or 1)
            labels_binary = torch.tensor([relabel_labels(label.item()) for label in labels]).to(device)
            
            # Sanity check: Print the relabeled labels for the first batch
           # if epoch == 0 and index == 0:  # Print for the first batch of the first epoch
               # print("Sanity check - labels (original):", labels)
                #print("Sanity check - labels (binary after relabeling):", labels_binary)
            
            output = model(images)
            loss = criterion(output, labels_binary.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            
            # Update loss and accuracy values for the training process
            train_loss += loss.item() * images.shape[0]
            prediction = (output > 0).float()
            equals = prediction == labels_binary.view(*prediction.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Relabel the validation labels into binary labels (0 or 1)
                labels_binary = torch.tensor([relabel_labels(label.item()) for label in labels]).to(device)
                
                # Sanity check: Print the relabeled labels for the first batch of validation data
                #if epoch == 0 and index == 0:  # Print for the first batch of the first epoch
                    #print("Sanity check - val labels (original):", labels)
                    #print("Sanity check - val labels (binary after relabeling):", labels_binary)
                
                output = model(images)
                loss = criterion(output, labels_binary.view(-1, 1).float())
                
                # Update loss and accuracy values for the validation process
                val_loss += loss.item() * images.shape[0]
                prediction = (output > 0).float()
                equals = prediction == labels_binary.view(*prediction.shape)
                val_acc += torch.mean(equals.type(torch.FloatTensor)).item()

        # Write training/validation loss/accuracy
        epoch_dict[epoch]['Train_Loss'] = train_loss / len(train_loader.sampler)
        epoch_dict[epoch]['Train_Accuracy'] = train_acc / len(train_loader)
        epoch_dict[epoch]['Val_Loss'] = val_loss / len(val_loader.sampler)
        epoch_dict[epoch]['Val_Accuracy'] = val_acc / len(val_loader)

        print(f"Epoch: {epoch+1}/{n_epochs}.. "
              f"Train Loss: {train_loss / len(train_loader.sampler):.3f}.. "
              f"Train Accuracy: {train_acc / len(train_loader):.3f}.. "
              f"Validation Loss: {val_loss / len(val_loader.sampler):.3f}.. "
              f"Validation Accuracy: {val_acc / len(val_loader):.3f}")
        
        # Save the model with the lowest validation loss
        if val_loss <= valid_loss_min:
            print('Validation loss decreased -- Saving model -- \n')
            torch.save(model.state_dict(), save_path)
            valid_loss_min = val_loss
    
    epoch_df = pd.DataFrame.from_dict(epoch_dict, orient='index')
    epoch_df['Epoch'] = np.arange(n_epochs) + 1
    epoch_df.to_csv(csv_path)  # Save as CSV (new, here!)
    return model, epoch_df

n_epochs = 50  #50
save_path = 'models/3BINARYe50.pth' #models/428BINARY.pth' 
csv_path = 'models/3BINARYe50_stats.csv'#models/epoch_results.csv' 
#save_path = 'models/428BINARY.pth'
#csv_path = 'models/epoch_results.csv' 

if os.path.exists(csv_path):
    epoch_df = pd.read_csv(csv_path)
else:
    print("CSV not found — was training skipped?")


batch = next(iter(train_loader))
images, labels = batch
images, labels = images.to(device), labels.to(device)
labels = labels.view(-1, 1).float()

#for _ in range(100):
    #optimizer.zero_grad()
    #output = model(images)
    #loss = criterion(output, labels)
    #loss.backward()
    #optimizer.step()
    #pred = (output > 0).float()
   # acc = (pred == labels).float().mean().item()
   # print("Labels:", labels.view(-1).cpu().numpy())
   # print("Predictions (after sigmoid):", torch.sigmoid(output).view(-1).detach().cpu().numpy())
   # print("Predicted Classes:", (output > 0).view(-1).cpu().numpy())
   ## print(f"Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
   # #break  # So it only prints once
    
#train model
#model, epoch_df = train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path)

#2.4 Visualize the loss and accuracy curves
epoch_df = pd.read_csv(csv_path) #before was csv_path
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
plt.savefig("testplot")
plt.show()

#2.5 Load Trained Model
model = Model()
state_dict = torch.load(save_path)
#if device.type == 'cpu':
 #   state_dict = torch.load(save_path, map_location='cpu')
#else:
    
model.load_state_dict(state_dict)
for param in model.parameters():
    param.requires_grad=False

#2.6 Load Images from test dataset
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

class_to_binary = {}
for idx, classname in enumerate(test_data.classes):
    if classname in cancerous_labels:
        class_to_binary[idx] = 1  # cancerous
    else:
        class_to_binary[idx] = 0  # noncancerous


print("Label to Binary Mapping:", class_to_binary)
print('Number of images in testing set: {}'.format(len(test_loader.sampler)))  #compared to 750 of original

#2.7 Make Prediction on test Dataset

def model_eval(test_loader, model, criterion, device, classes, class_to_binary):
    '''Apply the trained model to testing dataset and summarize the overall test_loss/test_acc
    also output testing accuracy for each category and the prediction result for each sample as well as the probability'''
    test_loss = 0.0
    test_acc = 0.0
    class_correct = [0.0, 0.0]
    class_total = [0.0, 0.0]

    dt = np.dtype(int)
    true_label = np.array([], dtype=dt)
    pred_label = np.array([], dtype=dt)
    path_array = np.array([])
    prob_array = np.array([])

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels, paths in test_loader:
            path_array = np.append(path_array, np.array(paths))
            images = images.to(device)
            labels = labels.to(device)

            # Convert multiclass labels to binary (0 or 1)
            labels_binary = torch.tensor([class_to_binary[int(label)] for label in labels], dtype=torch.float32).to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels_binary)
            test_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_tensor = preds.eq(labels_binary)
            correct = correct_tensor.sum().item()
            test_acc += correct
            print(test_acc)

            for i in range(len(labels_binary)):
                label = labels_binary[i].item()
                class_correct[int(label)] += correct_tensor[i].item()
                class_total[int(label)] += 1

            true_label = np.append(true_label, labels_binary.cpu().numpy())
            pred_label = np.append(pred_label, preds.cpu().numpy())
            prob_array = np.append(prob_array, torch.sigmoid(outputs).cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    print(test_acc)
    print(len(test_loader.dataset))

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc*100:.2f}% ({int(sum(class_correct))}/{int(sum(class_total))})')
    print('Test Accuracy of NONCANCEROUS: {:.3f}% ({}/{})'.format(
        100 * class_correct[0] / class_total[0], int(class_correct[0]), int(class_total[0])
    ))
    print('Test Accuracy of CANCEROUS: {:.3f}% ({}/{})'.format(
        100 * class_correct[1] / class_total[1], int(class_correct[1]), int(class_total[1])
    ))

    summary_df = pd.DataFrame({
        'class': ['NONCANCEROUS', 'CANCEROUS'],
        'correct': class_correct,
        'total': class_total
    })

    pred_df = pd.DataFrame({
        'file_path': path_array,
        'true_label': true_label,
        'prediction': pred_label,
        'prob': prob_array
    })

    return summary_df, pred_df



criterion = nn.BCEWithLogitsLoss()
summary_df, pred_df = model_eval(test_loader, model, criterion, device, classes, class_to_binary)
classes = ["NONCANCEROUS", "CANCEROUS"]
print(summary_df)
#skipped a CSV saving
summary_df['accuracy'] = np.round(summary_df.correct/summary_df.total*100,2)
fig = plt.figure(figsize = (10,4))
ax = fig.add_subplot(1,1,1)
width = 0.4 # the width of the bars 
ind = np.arange(len(classes))  # the x locations for the groups
ax.bar(ind, summary_df.accuracy, width, edgecolor='black', color = sns.color_palette('hls', 2))
plt.xticks(ind, labels = classes, fontsize=18, fontweight='bold')
plt.yticks(fontsize=16)
plt.xlabel('Category', fontsize=18)
plt.ylabel('Accuracy%', fontsize=18)

for i, v in enumerate(summary_df.accuracy):
    ax.text(i-0.25, v + 1, str(np.round(v,2))+'%', fontweight='bold', fontsize=18)
ax.plot([-0.5, 1.5], [94.815, 94.815], "k--", linewidth=2)
ax.text(1.55, 94.815, 'Avg:\n'+str(94.815)+'%', fontweight='bold', fontsize=18)
plt.show()


#2.8 Confusion Matrix
# Filter the data to only include 'cancerous' (1) and 'noncancerous' (0)
pred_df_filtered = pred_df[pred_df['true_label'].isin([0, 1])]  # 0: noncancerous, 1: cancerous

# Apply binary remapping to both the true labels and the predicted labels AFTER filtering
pred_df_filtered['true_bin'] = pred_df_filtered.true_label.map(class_to_binary)
pred_df_filtered['pred_bin'] = pred_df_filtered.prediction.map(class_to_binary)

# Get the true and predicted values
y_true = pred_df_filtered.true_bin.values
y_pred = pred_df_filtered.pred_bin.values

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):    
    plt.figure(figsize=(10,7))  # Smaller figure size
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)  # Reduce font size of title
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, fontweight='bold')  # Smaller font size for ticks
    plt.yticks(tick_marks, classes, fontsize=10, fontweight='bold')  # Smaller font size for ticks

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:d} \n({:.2f}%)'.format(cm[i, j], cm[i,j]/cm.sum()*100), 
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=20)  # Smaller font for text in cells

    plt.ylabel('True label', fontsize=12)  # Smaller font for label
    plt.xlabel('Predicted label', fontsize=12)  # Smaller font for label
    
    # Adjust layout with more padding
    plt.tight_layout(pad=4.0)  # Reduce padding to make everything a bit tighter
    plt.subplots_adjust(right=0.85, top=0.88)  # Keep margins adjusted for smaller size

    plt.show()

# Classes for the confusion matrix (noncancerous, cancerous)
classes = ['NONCANCEROUS', 'CANCEROUS']

# Plot the confusion matrix for cancerous and noncancerous
plot_confusion_matrix(cm, classes, title='Confusion Matrix for Cancerous vs Noncancerous')


#ROC Curve
pred_df.head()
# Compute FPR, TPR, and AUC
fpr, tpr, _ = roc_curve(pred_df['true_label'], pred_df['prob'])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Binary Classification ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()