from matplotlib import pyplot as plt
import pandas as pd

epoch_dfMulti1 = pd.read_csv("models/multiclass_stats.csv")
epoch_dfMulti2 = pd.read_csv("models/2multiclass_stats.csv")
epoch_dfMulti3 = pd.read_csv("models/3multiclass_stats.csv")
epoch_dfBinary1 = pd.read_csv("models/noBINARYe50_stats.csv")
epoch_dfBinary2 = pd.read_csv("models/2BINARYe50_stats.csv")
epoch_dfBinary3 = pd.read_csv("models/3BINARYe50_stats.csv")

multiValLosses = [epoch_dfMulti1.Val_Loss, epoch_dfMulti2.Val_Loss, epoch_dfMulti3.Val_Loss]
multiAvgTrain_Loss = sum(multiValLosses)/len(multiValLosses)
print(multiAvgTrain_Loss)

'''
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(50, epoch_df.Train_Loss, label = 'Training loss')
plt.plot(50, epoch_df.Val_Loss, label = 'Validation loss')
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
    #Apply the trained model to testing dataset and summarize the overall test_loss/test_acc
    #also output testing accuracy for each category and the prediction result for each sample as well as the probability
    
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
    print(f"{class_name}: {auc_value:.2f}")'''