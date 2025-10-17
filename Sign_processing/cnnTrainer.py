import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Sign_processing.cnn import CNN



#This class handles the training of the CNN model
#It loads training data, trains the model, and saves the weights
class CNNTrainer:
    def __init__(self, train_dir, model_path, classes_path):
        self.train_dir = train_dir
        self.model_path = model_path
        self.classes_path = classes_path

        self.img_size = 128
        self.batch_size =  128
        self.epochs = 6
        self.lr = 1e-3
        self.num_workers = 0
        self.allowed_exts = {".png", ".jpg", ".jpeg"}

    def get_transforms(self):
        train_tfm = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5)),
        ])
        return train_tfm

    def is_allowed_file(self, path):
        return Path(path).suffix.lower() in self.allowed_exts

    def build_dataset(self):
        train_tfm = self.get_transforms()
        train_ds = datasets.ImageFolder(str(self.train_dir), transform=train_tfm, is_valid_file=self.is_allowed_file)
        return train_ds

    def train(self):
        train_ds = self.build_dataset()

        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=pin
        )

        num_classes = len(train_ds.classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        lr=self.lr        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, self.epochs + 1):
            model.train()
            run_loss = 0
            correct = 0
            total = 0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = run_loss / max(1, total)
            train_acc = correct / max(1, total)


            
            torch.save(model.state_dict(), self.model_path)
            with open(self.classes_path, "w") as f:
                json.dump({"classes": train_ds.classes}, f)

        torch.save(model.state_dict(), self.model_path)
        with open(self.classes_path, "w") as f:
            json.dump({"classes": train_ds.classes}, f)

        
if __name__ == "__main__":
    root = Path(__file__).parent / "demo"
    train_dir = root / "train"
    model_path = root / "cnn.pth"
    classes_path = root / "classes.json"

    if len(sys.argv) < 2:
        print("Usage: python CNNTrainer.py train")
        sys.exit(0)

    if sys.argv[1] == "train":
        trainer = CNNTrainer(train_dir, model_path, classes_path)
        trainer.train()
    else:
        print("Unknown command.")