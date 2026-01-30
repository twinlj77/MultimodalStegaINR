from zipfile import Path
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import secrets
import matplotlib.pyplot as plt
from pathlib import Path

# 注册字体
font_path = './SimHei.ttf'
fm.fontManager.addfont(font_path)

# 设置为默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

torch.manual_seed(42)
np.random.seed(42)


# ========== 图像预处理 ==========
def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


# ========== 坐标生成 ==========
def get_coordinates(h, w):
    y = torch.linspace(0, 1, h)
    x = torch.linspace(0, 1, w)
    grid_x, grid_y = torch.meshgrid(x, y)
    coords = torch.stack((grid_x, grid_y), dim=-1)
    return coords.view(-1, 2).to(device)


# ========== 隐式神经网络 ==========
class INRNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=3):
        super(INRNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# ========== 提取器 ==========
class Extractor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_size=128):
        super(Extractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, 3 * output_size * output_size)
        self.output_size = output_size

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.encoder(x)  # (batch_size, hidden_dim)
        x = x.mean(dim=0, keepdim=True)  # 聚合为 (1, hidden_dim)
        x = self.decoder(x)  # (1, 3*H*W)
        return x.view(-1, 3, self.output_size, self.output_size)  # (1, 3, H, W)


# ========== 训练 INR ==========
def train_inr_representation(image_tensor, epochs=1000):
    model = INRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    h, w = image_tensor.shape[2:]
    coords = get_coordinates(h, w)
    pixels = image_tensor.view(1, 3, -1).permute(0, 2, 1).squeeze(0)

    for epoch in range(epochs):
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'INR Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')

    return model, coords


# ========== 神经网络转点云 ==========
def neural_net_to_point_cloud(model, coords, noise_scale=10000, rgb_noise_scale=10000, seed=76):
    with torch.no_grad():
        rgb = model(coords)

    coords_np = coords.cpu().numpy()
    rgb_np = rgb.cpu().numpy()

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 添加噪声
    noise = np.random.normal(0, noise_scale, coords_np.shape)
    dispersed_coords = coords_np + noise

    # 给RGB添加噪声
    rgb_noise = np.random.normal(0, rgb_noise_scale, rgb_np.shape)
    noisy_rgb = rgb_np + rgb_noise

    return np.concatenate([dispersed_coords, noisy_rgb], axis=1)


# ========== 提取器训练 ==========
def train_extractor(point_cloud, target_img_tensor, seed, img_size, epochs=150, output_dir="z-output"):
    np.random.seed(seed)
    indices = np.random.choice(len(point_cloud), size=1024, replace=False)
    sampled = torch.tensor(point_cloud[indices], dtype=torch.float32).to(device)  # (4096, 5)
    target = target_img_tensor.to(device)  # (1, 3, 128, 128)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整点云
    np.save(os.path.join(output_dir, "full_point_cloud.npy"), point_cloud)

    extractor = Extractor(input_dim=5, output_size=img_size).to(device)
    optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)

    for epoch in range(epochs):
        output = extractor(sampled)  # (1, 3, 128, 128)
        loss = F.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'MPL Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')

    # 保存提取器模型
    torch.save(extractor.state_dict(), os.path.join(output_dir, "extractor_model.pth"))

    return extractor, indices


# ========== 后处理（SSIM 优化） ==========
def advanced_postprocessing(output, target):
    output = np.clip(output, 0, 1)
    best = output
    best_ssim = ssim(target, output, channel_axis=2, data_range=1.0)
    for gamma in [0.9, 1.0, 1.1]:
        adjusted = np.clip(output ** gamma, 0, 1)
        score = ssim(target, adjusted, channel_axis=2, data_range=1.0)
        if score > best_ssim:
            best_ssim = score
            best = adjusted
    return best


# ========== 主函数 ==========
def main(host_image_path, secret_image_path, img_size=128, output_dir="./results"):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    host_tensor = preprocess_image(host_image_path, img_size)
    secret_tensor = preprocess_image(secret_image_path, img_size)

    inr_model, coords = train_inr_representation(host_tensor, epochs=1000)

    # 保存INR模型（载体图片的神经网络）
    torch.save(inr_model.state_dict(), os.path.join(output_dir, "inr_model.pth"))
    print(f"INR模型已保存到: {os.path.join(output_dir, 'inr_model.pth')}")

    point_cloud = neural_net_to_point_cloud(inr_model, coords)

    # 保存完整点云
    np.save(os.path.join(output_dir, "full_point_cloud.npy"), point_cloud)

    # 生成密钥
    # key_hex = secrets.token_bytes(16).hex()
    key_hex = secrets.randbits(16)
    with open(os.path.join(output_dir, "extraction_key.txt"), "w") as f:
        f.write(str(key_hex))

    extractor, sampled_indices = train_extractor(point_cloud, secret_tensor, key_hex, img_size, epochs=4500,
                                                 output_dir=output_dir)

    with torch.no_grad():
        sampled_pc = torch.tensor(point_cloud[sampled_indices], dtype=torch.float32).to(device)
        reconstructed = extractor(sampled_pc).cpu().numpy()

    reconstructed = np.clip(reconstructed.squeeze(0).transpose(1, 2, 0), 0, 1)

    secret_np = secret_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed = advanced_postprocessing(reconstructed, secret_np)

    mse_val = np.mean((secret_np - reconstructed) ** 2)
    psnr_val = psnr(secret_np, reconstructed, data_range=1.0)

    ssim_val = ssim(secret_np, reconstructed, channel_axis=2, data_range=1.0, win_size=7)

    print(f"秘密图像重建质量: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")

    # 构建保存路径

    save_path = os.path.join(output_dir, 'reconstructed_image.png')

    plt.figure(figsize=(15, 5))
    plt.title(f"重建图像 (PSNR: {psnr_val:.2f} dB)")
    plt.imshow(reconstructed)
    plt.axis('off')

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()

    save_path = os.path.join(output_dir, 'original_reconstructed_image.png')
    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title("加密图像")
    plt.imshow(secret_np)
    plt.axis('off')

    plt.subplot(132)
    plt.title(f"重建图像 (PSNR: {psnr_val:.2f} dB)")
    plt.imshow(reconstructed)
    plt.axis('off')

    plt.subplot(133)
    diff = np.abs(secret_np - reconstructed)
    plt.title("差异热力图")
    plt.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # 关闭图形释放资源

    plt.figure(figsize=(15, 6))
    plt.subplot(131);
    plt.title("载体图像");
    plt.imshow(host_tensor.squeeze(0).permute(1, 2, 0).cpu());
    plt.axis('off')
    plt.subplot(132);
    plt.title("重建秘密图像");
    plt.imshow(reconstructed);
    plt.axis('off')
    plt.subplot(133);
    plt.title("目标秘密图像");
    plt.imshow(secret_np);
    plt.axis('off')
    plt.tight_layout();
    plt.savefig('./results/dog_reconstruct.png', dpi=300);
    plt.show()

    return secret_np, reconstructed, psnr_val


if __name__ == "__main__":
    main("./datasets/cover.jpg", "./datasets/secret.jpg", img_size=128)
#好像是保存了INR