# Put model on device SANITY CHECK
model = model.to(device)
model.train()

# Get a small batch
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

# ----> Relabel to binary (very important)
labels_binary = (labels == 0) | (labels == 2) | (labels == 3)
labels_binary = labels_binary.float()
labels_binary = labels_binary.view(-1, 1)  # <<< fix shape

# Set up optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    # Create binary labels
    labels_binary = (labels == 0) | (labels == 2) | (labels == 3)
    labels_binary = labels_binary.float()
    labels_binary = labels_binary.view(-1, 1)

    # --- New: Print the original and binary labels ---
    print("Original labels:", labels.cpu().numpy())
    print("Binary labels  :", labels_binary.squeeze().cpu().numpy())
    break  # Only do this for the first batch
# Train for a few steps on this single batch
for step in range(20):
    optimizer.zero_grad()

    output = model(images)

    # Calculate loss
    loss = criterion(output, labels_binary)

    # Backprop
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    preds = (output > 0).float()
    acc = (preds == labels_binary.view_as(preds)).float().mean()

    print(f"Step {step+1} - Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")




    #training 
    #train goes through dataset for n_epochs, determining the best model w/ train_acc, train_loss, val_acc and val_loss. // saves the most effective
def train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path):
    '''the train function will perform both forward and backpropagation on training and validation datasets.
    Output: trained model with the lowest val_loss and dataframe containing train_loss/train_acc and val_loss/val_acc for each epoch'''
    epoch_dict={}
    valid_loss_min = np.inf
    model = model.to(device)
    for index, epoch in enumerate(np.arange(n_epochs)):  #added enumerate, and index
        print("Epoch: {}, Value from arange: {}".format(epoch, index))
        # Initiate loss and accuracy values
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        epoch_dict[epoch] = {}
        # Training
        model.train()
        
        a = 0 #added a for tracking
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"): #trainloader
            #print(labels)
           
            labels = torch.where((labels == 0) | (labels == 2) | (labels == 3), 
            torch.tensor(1.0, device=labels.device), 
            torch.tensor(0.0, device=labels.device))
            labels = labels.view(-1, 1).float()
            
            #print(labels)
            
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
    
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            ## Update loss and acc values for the training process
            train_loss += loss.item()*images.shape[0]
            prediction = (output > 0).float()
            equals = prediction == labels.view(*prediction.shape)
            #train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            train_acc += torch.sum(equals).item()
            print(len(train_loader)-a)
            a+=1

        # Validation
        else:
            # turn off gradients
            with torch.no_grad():
                # set model to evaluation mode
                model.eval()
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels = labels.view(-1, 1).float()
                    output = model(images)
                    loss = criterion(output, labels)
                    ## Update loss and acc values for the validation process
                    val_loss += loss.item()*images.shape[0]
                     # Binary classification: apply threshold of 0.5 to get predictions
                    prediction = (output > 0).float()  # Logit > 0 is class 1, otherwise class 0
                    equals = prediction == labels.view(*prediction.shape)  # Compare predictions to labels
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




#train with sanity check
def train(n_epochs, train_loader, val_loader, model, criterion, optimizer, device, save_path):
    '''the train function will perform both forward and backpropagation on training and validation datasets.
    Output: trained model with the lowest val_loss and dataframe containing train_loss/train_acc and val_loss/val_acc for each epoch'''
    
    epoch_dict = {}
    valid_loss_min = np.inf
    model = model.to(device)
    
    # Sanity check: Print labels to ensure they are binary
    print("Sanity check - Checking labels in the training data:")
    for images, labels in train_loader:
        print("Labels:", set(labels.cpu().numpy()))  # Print the unique values in the labels
        break  # Check only for the first batch

    for index, epoch in enumerate(np.arange(n_epochs)): #added enumerate, and index
        print(f"Epoch: {epoch}, Value from arange: {index}")
        # Initiate loss and accuracy values
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        epoch_dict[epoch] = {}
        # Training
        model.train()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"): #trainloader
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            
            # Apply binary relabeling
            labels = torch.where((labels == 0) | (labels == 2) | (labels == 3), torch.tensor(1).to(device), torch.tensor(0).to(device))
            
            labels = labels.view(-1, 1).float()  # Ensure labels are in correct shape for binary classification
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Update loss and acc values for the training process
            train_loss += loss.item() * images.shape[0]
            prediction = (output > 0).float()  # Apply threshold for binary classification
            equals = prediction == labels.view(*prediction.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor)).item()

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Apply binary relabeling
                labels = torch.where((labels == 0) | (labels == 2) | (labels == 3), torch.tensor(1).to(device), torch.tensor(0).to(device))
                
                labels = labels.view(-1, 1).float()  # Ensure labels are in correct shape for binary classification
                output = model(images)
                loss = criterion(output, labels)

                # Update loss and acc values for the validation process
                val_loss += loss.item() * images.shape[0]
                prediction = (output > 0).float()  # Apply threshold for binary classification
                equals = prediction == labels.view(*prediction.shape)
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

        # Save the model with the lowest val_loss
        if val_loss <= valid_loss_min:
            print('Validation loss decreased -- Saving model -- \n')
            torch.save(model.state_dict(), save_path)
            valid_loss_min = val_loss

    epoch_df = pd.DataFrame.from_dict(epoch_dict, orient='index')
    epoch_df['Epoch'] = np.arange(n_epochs) + 1
    epoch_df.to_csv(csv_path)  # Save as CSV
    return model, epoch_df







#2.7 Make Prediction on test Dataset
def model_eval(test_loader, model, criterion, device, classes, class_to_binary):
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
        class_correct = [0.0, 0.0]
        class_total = [0.0, 0.0]
        for images, labels, paths in test_loader:
            path_array = np.append(path_array, np.array(paths))
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            output = output.squeeze(1)
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
                remapped_label = class_to_binary[label]
                class_correct[remapped_label] += correct[i].item()
                #class_correct[label] += correct[i].item()
                class_total[remapped_label] += 1
    
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
    
    print('Test Accuracy of {}: {:.3f}% ({}/{})'.format("NONCANCEROUS", class_correct[0]/class_total[0]*100,class_correct[0], class_total[0]))
    print('Test Accuracy of {}: {:.3f}% ({}/{})'.format("CANCEROUS", class_correct[1]/class_total[1]*100,class_correct[1], class_total[1]))
    
    #summary_df = pd.DataFrame({'category': classes, 'correct': class_correct, 'total': class_total})
    summary_df = pd.DataFrame({
    'class': ['NONCANCEROUS', 'CANCEROUS'],
    'correct': class_correct,
    'total': class_total
    })
    pred_df = pd.DataFrame({'file_path': path_array, 'true_label': true_label, 'prediction': pred_label, 'prob': prob_array})
    probs_df = pd.DataFrame(probs_array[1:], columns = probs_array[0])
    probs_df['file_path'] = path_array
    pred_df = pred_df.merge(probs_df, left_on = 'file_path', right_on = 'file_path')
    return summary_df, pred_df