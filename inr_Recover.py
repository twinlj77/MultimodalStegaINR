# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ========== INR网络定义 ==========
class INRNet(torch.nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=3):
        super(INRNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# ========== 提取器定义 ==========
class Extractor(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_size=128):
        super(Extractor, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, 3 * output_size * output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=0, keepdim=True)
        x = self.decoder(x)
        return x.view(-1, 3, self.output_size, self.output_size)


# ========== 坐标生成 ==========
def get_coordinates(h, w):
    y = torch.linspace(0, 1, h)
    x = torch.linspace(0, 1, w)
    grid_x, grid_y = torch.meshgrid(x, y)
    coords = torch.stack((grid_x, grid_y), dim=-1)
    return coords.view(-1, 2).to(device)


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


# ========== 提取秘密图像 ==========
def extract_secret_image(output_dir, img_size=128):
    # 加载INR模型
    inr_model = INRNet().to(device)
    inr_model.load_state_dict(torch.load(os.path.join(output_dir, "inr_model.pth"), map_location=device))
    inr_model.eval()

    # 生成坐标
    coords = get_coordinates(img_size, img_size)

    # 生成完整点云
    point_cloud = neural_net_to_point_cloud(inr_model, coords)
    print(f"生成的完整点云形状: {point_cloud.shape}")

    # 加载密钥
    with open(os.path.join(output_dir, "extraction_key.txt"), "r") as f:
        secret_key = int(f.read().strip())

    # 使用密钥生成秘密点云
    np.random.seed(secret_key)
    indices = np.random.choice(len(point_cloud), size=1024, replace=False)
    secret_point_cloud = point_cloud[indices]
    print(f"采样的秘密点云形状: {secret_point_cloud.shape}")

    # 保存秘密点云
    np.save(os.path.join(output_dir, "secret_point_cloud.npy"), secret_point_cloud)
    print(f"秘密点云已保存到: {os.path.join(output_dir, 'secret_point_cloud.npy')}")

    # 加载提取器模型
    extractor = Extractor(input_dim=5, output_size=img_size)
    extractor.load_state_dict(torch.load(os.path.join(output_dir, "extractor_model.pth"), map_location=device))
    extractor.to(device)
    extractor.eval()

    # 使用提取器重建秘密图像
    with torch.no_grad():
        sampled_pc = torch.tensor(secret_point_cloud, dtype=torch.float32).to(device)
        reconstructed = extractor(sampled_pc).cpu().numpy()

    # 后处理
    reconstructed = np.clip(reconstructed.squeeze(0).transpose(1, 2, 0), 0, 1)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    # 保存提取的秘密图像
    output_path = os.path.join(output_dir, 'extracted_secret.png')
    Image.fromarray(reconstructed).save(output_path)
    print(f"提取的秘密图像已保存到: {output_path}")

    # 显示提取的秘密图像
    plt.imshow(reconstructed)
    plt.title("Extracted Secret Image")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    output_dir = './results'  # 包含INR模型、提取器模型和密钥的目录
    extract_secret_image(output_dir)
