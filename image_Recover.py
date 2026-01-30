# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
import secrets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


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
        x = self.encoder(x)
        x = x.mean(dim=0, keepdim=True)
        x = self.decoder(x)
        return x.view(-1, 3, self.output_size, self.output_size)


# ========== 训练 INR ==========
def train_inr_representation(image_tensor, epochs=1000):
    model = INRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    h, w = image_tensor.shape[2:]
    coords = get_coordinates(h, w)
    pixels = image_tensor.view(1, 3, -1).permute(0, 2, 1).squeeze(0)

    for epoch in range(epochs):
        pred = model(coords)
        loss = nn.functional.mse_loss(pred, pixels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'INR Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')

    return model, coords


# ========== 生成秘密点云 ==========
def generate_secret_point_cloud(full_point_cloud, secret_key, sample_size=1024):
    # 使用密钥生成采样索引
    np.random.seed(secret_key)
    indices = np.random.choice(len(full_point_cloud), size=sample_size, replace=False)
    secret_point_cloud = full_point_cloud[indices]
    return secret_point_cloud, indices


# ========== 提取秘密图像 ==========
def extract_secret_image(extractor_model_path, secret_point_cloud, img_size=128):
    # 加载提取器模型
    extractor = Extractor(input_dim=5, output_size=img_size)
    extractor.load_state_dict(torch.load(extractor_model_path, map_location=device))
    extractor.to(device)
    extractor.eval()

    # 转换为张量
    secret_pc_tensor = torch.tensor(secret_point_cloud, dtype=torch.float32).to(device)

    # 使用提取器重建秘密图像
    with torch.no_grad():
        reconstructed = extractor(secret_pc_tensor).cpu().numpy()

    # 后处理
    reconstructed = np.clip(reconstructed.squeeze(0).transpose(1, 2, 0), 0, 1)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    return reconstructed


# ========== 主函数 ==========
def main(cover_image_path, output_dir="./results", img_size=128):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 预处理载体图像
    print("预处理载体图像...")
    host_tensor = preprocess_image(cover_image_path, img_size)

    # 2. 训练INR模型
    print("训练INR模型...")
    inr_model, coords = train_inr_representation(host_tensor, epochs=1000)

    # 保存INR模型
    inr_model_path = os.path.join(output_dir, "inr_model.pth")
    torch.save(inr_model.state_dict(), inr_model_path)
    print(f"INR模型已保存到: {inr_model_path}")

    # 3. 生成完整点云
    print("生成完整点云...")
    full_point_cloud = neural_net_to_point_cloud(inr_model, coords)

    # 保存完整点云
    full_pc_path = os.path.join(output_dir, "full_point_cloud.npy")
    np.save(full_pc_path, full_point_cloud)
    print(f"完整点云已保存到: {full_pc_path}")

    # 4. 读取密钥
    key_path = os.path.join(output_dir, "extraction_key.txt")
    if not os.path.exists(key_path):
        print("未找到密钥文件，生成新密钥...")
        secret_key = secrets.randbits(16)
        with open(key_path, "w") as f:
            f.write(str(secret_key))
    else:
        with open(key_path, "r") as f:
            secret_key = int(f.read().strip())
    print(f"使用密钥: {secret_key}")

    # 5. 生成秘密点云
    print("生成秘密点云...")
    secret_point_cloud, indices = generate_secret_point_cloud(full_point_cloud, secret_key)

    # 保存秘密点云
    secret_pc_path = os.path.join(output_dir, "secret_point_cloud.npy")
    np.save(secret_pc_path, secret_point_cloud)
    print(f"秘密点云已保存到: {secret_pc_path}")

    # 6. 提取秘密图像
    print("提取秘密图像...")
    extractor_model_path = os.path.join(output_dir, "extractor_model.pth")
    if not os.path.exists(extractor_model_path):
        print("错误: 未找到提取器模型文件!")
        return

    reconstructed = extract_secret_image(extractor_model_path, secret_point_cloud, img_size)

    # 7. 保存提取的秘密图像
    output_path = os.path.join(output_dir, 'extracted_secret.png')
    Image.fromarray(reconstructed).save(output_path)
    print(f"提取的秘密图像已保存到: {output_path}")

    # 8. 显示提取的秘密图像
    plt.imshow(reconstructed)
    plt.title("提取的秘密图像")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    cover_image_path = "./datasets/cover.jpg"
    output_dir = "./results"
    main(cover_image_path, output_dir)