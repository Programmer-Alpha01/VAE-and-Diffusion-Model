import torch

class DiffusionReverseProcess:
    
    def __init__(self, 
                 num_time_steps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02):
        
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Precompute for speed
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recip_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bars - 1)

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Sample x_{t-1} from x_t + predicted noise.
        t can be int, scalar tensor, or tensor of shape (B,)
        """
        device = xt.device
        B = xt.shape[0]
        
        # === Clean timestep handling ===
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.long)
        else:
            t = t.to(device=device, dtype=torch.long)
        
        # Make sure t has shape (B,) 
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        elif t.shape[0] != B:
            t = t[:B]  # safety
        
        # Get values at timestep t (broadcasted)
        alpha_bar_t = self.alpha_bars[t].to(device).view(-1, 1, 1, 1)
        alpha_t     = self.alphas[t].to(device).view(-1, 1, 1, 1)
        beta_t      = self.betas[t].to(device).view(-1, 1, 1, 1)
        
        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Compute mean for x_{t-1}
        mean = (xt - (1 - alpha_t) * noise_pred / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
        
        # For t == 0, return clean x0 (no noise)
        if torch.all(t == 0):
            return mean, x0_pred
        
        # Add variance for t > 0
        alpha_bar_tm1 = self.alpha_bars[t-1].to(device).view(-1, 1, 1, 1)
        variance = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
        sigma = torch.sqrt(variance)
        
        z = torch.randn_like(xt)
        
        return mean + sigma * z, x0_pred


if __name__ == "__main__":
    # Test
    xt = torch.randn(1, 3, 256, 256)
    noise_pred = torch.randn(1, 3, 256, 256)
    t = 500
    drp = DiffusionReverseProcess()
    out, x0 = drp.sample_prev_timestep(xt, noise_pred, t)
    print("Output shape:", out.shape)