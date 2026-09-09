from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class NormalImageDataset(Dataset):
    def __init__(self, folder, image_size):
        self.files = sorted(Path(folder).glob("*.png"))
        self.image_size = tuple(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("L")
        img = img.resize(self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


def main():
    with open("configs/anomaly.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["autoencoder"]

    image_size = cfg["input_size"]

    train_ds = NormalImageDataset("data/anomaly_ae/train", image_size)
    val_ds = NormalImageDataset("data/anomaly_ae/val", image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = ConvAutoencoder(cfg["latent_dim"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )
    criterion = nn.MSELoss()

    output = Path("models/anomaly/autoencoder.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print("Device: cpu")
    print(f"Epochs: {cfg['epochs']}")

    best_val_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_total = 0.0

        for images in train_loader:
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)

        train_loss = train_total / len(train_ds)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images in val_loader:
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_total += loss.item() * images.size(0)

        val_loss = val_total / len(val_ds)

        print(
            f"Epoch {epoch + 1:02d}/{cfg['epochs']} "
            f"- train={train_loss:.6f} "
            f"- val={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "latent_dim": cfg["latent_dim"],
                    "input_size": cfg["input_size"],
                    "architecture": cfg["architecture"],
                    "threshold": cfg["reconstruction_threshold"],
                },
                output,
            )

    print(f"Saved model: {output}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
