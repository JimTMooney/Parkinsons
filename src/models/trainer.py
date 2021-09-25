import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix

class Trainer():
    def __init__(self, dataset, weight_decay, n_epochs, lr=1e-3, log = False):
        self.dataset = dataset
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.lr = lr
        self.log = log
        
        
    # Complete this later
    def find_lr(self):
        pass
    
    def full_batch_loop(self, net, train_loader, optimizer, criterion):
        n_train=0
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            output = net(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += (loss.item() * len(labels))

            preds = torch.argmax(output, axis=1)
            train_correct += sum(preds == labels).item()
            n_train += len(labels)


        train_accuracy = train_correct / n_train
        
        return train_loss, train_accuracy
    
    def one_element_loop(self, net, train_loader, optimizer, criterion):
        n_train = 0
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            for idx, img in enumerate(imgs):
                label = labels[idx]
                output = net(img)
                loss = criterion(output, label)
                loss.backward()
                train_loss += loss.item()
                pred = torch.argmax(output, axis=1)
                train_correct += sum(pred == label).item()
                n_train += 1
            optimizer.step()
        
        train_accuracy = train_correct / n_train 
        
        return train_loss, train_accuracy
        
        
    def train(self, train_loader, net, test_loader=None, full_batch=True):
        net.train()
        self.dataset.change_mode(train_mode=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        
        n_train = 0
        for epoch in range(self.n_epochs):
            if full_batch:
                train_loss, train_accuracy = self.full_batch_loop(net, train_loader, optimizer, criterion)
            else:
                train_loss, train_accuracy = self.one_element_loop(net, train_loader, optimizer, criterion)
                
            
            print('epoch %d train (loss, accuracy) ----> (%.4e, %.4e)' % (epoch, train_loss, train_accuracy))
            
            if test_loader is not None:
                _, test_accuracy, test_loss = self.test_metrics(test_loader, net, full_batch=full_batch)
                print('epoch %d test (loss, accuracy) ----> (%.4e, %.4e)\n' % (epoch, test_loss, test_accuracy))
            
    
    def test_loop(self, test_loader, net, criterion, n_crop=False, full_batch=True):
        conf_mat = torch.zeros((2, 2))
        test_loss = 0.0
        if full_batch:
            for imgs, labels in test_loader:
                outputs = net(imgs)
                test_loss += (criterion(outputs, labels).item() * len(labels))
                preds = torch.argmax(outputs, axis=1)
                conf_mat += confusion_matrix(labels.cpu(), preds.cpu())
        else:
            for imgs, labels in test_loader:
                for idx, img in enumerate(imgs):
                    label = labels[idx]
                    output = net(img)
                    test_loss += criterion(output, label).item()
                    pred = torch.argmax(output, axis=1)
                    conf_mat += confusion_matrix(label.cpu(), pred.cpu())
                    
        return conf_mat, test_loss
        
      
        
    def test_metrics(self, test_loader, net, n_crop=False, full_batch=True):
        net.eval()
        self.dataset.change_mode(train_mode=False)
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            conf_mat, test_loss = self.test_loop(test_loader, net, criterion, n_crop=n_crop, full_batch=full_batch)
                
        net.train()
        self.dataset.change_mode(train_mode=True)
        
        accuracy = (conf_mat[0, 0] + conf_mat[1, 1]).item() / torch.sum(conf_mat).item()
        return conf_mat, accuracy, test_loss
            
        
    # Complete later
    def swa_train(self):
        pass
        
    # Complete later
    def save_model(self):
        pass