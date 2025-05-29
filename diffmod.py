import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, down_channels=(64, 128, 256), up_channels=(256, 128, 64), time_emb_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        
        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) 
                                    for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) 
                                  for i in range(len(up_channels)-1)])
        
        # Final conv
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

class DiffusionModel:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, x_0.device)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, x, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, image_size, batch_size=1, channels=3):
        device = next(self.model.parameters()).device
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        for i in range(0, self.timesteps)[::-1]:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t)
            
        return img

def load_image(image_path, image_size=64):
    """Load and preprocess a single image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def visualize_reconstruction_steps(model, diffusion, original_image, epoch, save_dir="training_viz", num_steps=10):
    """Visualize the reconstruction process at different noise levels."""
    model.eval()
    device = original_image.device
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Select timesteps to visualize
    timesteps = torch.linspace(0, diffusion.timesteps-1, num_steps).long()
    
    fig, axes = plt.subplots(2, num_steps, figsize=(20, 8))
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Forward process: add noise to original image
            t_tensor = torch.tensor([t], device=device)
            noisy_image, noise = diffusion.forward_diffusion_sample(original_image, t_tensor, device)
            
            # Backward process: predict noise
            predicted_noise = model(noisy_image, t_tensor)
            
            # Denormalize for visualization
            noisy_vis = (noisy_image.squeeze(0).cpu() + 1) / 2
            noisy_vis = torch.clamp(noisy_vis, 0, 1)
            
            # Calculate denoised image
            betas_t = diffusion.get_index_from_list(diffusion.betas, t_tensor, noisy_image.shape)
            sqrt_one_minus_alphas_cumprod_t = diffusion.get_index_from_list(
                diffusion.sqrt_one_minus_alphas_cumprod, t_tensor, noisy_image.shape
            )
            sqrt_recip_alphas_t = diffusion.get_index_from_list(diffusion.sqrt_recip_alphas, t_tensor, noisy_image.shape)
            
            denoised = sqrt_recip_alphas_t * (
                noisy_image - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )
            denoised_vis = (denoised.squeeze(0).cpu() + 1) / 2
            denoised_vis = torch.clamp(denoised_vis, 0, 1)
            
            # Plot noisy image
            axes[0, i].imshow(noisy_vis.permute(1, 2, 0))
            axes[0, i].set_title(f't={t}')
            axes[0, i].axis('off')
            
            # Plot denoised prediction
            axes[1, i].imshow(denoised_vis.permute(1, 2, 0))
            axes[1, i].set_title(f'Denoised')
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Noisy Images', rotation=90, size='large')
    axes[1, 0].set_ylabel('Predicted Denoised', rotation=90, size='large')
    
    plt.suptitle(f'Reconstruction Steps - Epoch {epoch}', size=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.train()

def train_on_single_image(image_path, epochs=1000, lr=1e-3, image_size=64, viz_every=100, save_viz=True):
    """Train the diffusion model on a single image."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    image = load_image(image_path, image_size).to(device)
    
    # Initialize model
    model = SimpleUNet().to(device)
    diffusion = DiffusionModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        t = torch.randint(0, diffusion.timesteps, (1,), device=device).long()
        loss = diffusion.get_loss(model, image, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Save visualization every N epochs
        if save_viz and epoch % viz_every == 0:
            visualize_reconstruction_steps(model, diffusion, image, epoch)
    
    return model, diffusion

def generate_samples(model, diffusion, num_samples=4, image_size=64):
    """Generate samples from the trained model."""
    model.eval()
    samples = diffusion.sample(image_size, batch_size=num_samples)
    
    # Denormalize samples
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples

# Example usage:
if __name__ == "__main__":
    # Example training (you need to provide an image path)
    model, diffusion = train_on_single_image("test_img.jpg")
    
    # Example generation
    # samples = generate_samples(model, diffusion)
    
    print("Diffusion model created! Use train_on_single_image() to train on your image.")
    print("Example: model, diffusion = train_on_single_image('path/to/your/image.jpg')")
