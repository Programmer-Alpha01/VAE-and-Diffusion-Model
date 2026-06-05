import torch
import torch.nn as nn

class DiffusionReverseProcess:
    
    r"""
    
    Reverse Process class as described in the 
    paper "Denoising Diffusion Probabilistic Models"
    
    """
    
    def __init__(self, 
                 num_time_steps = 1000, 
                 beta_start = 1e-4, 
                 beta_end = 0.02
                ):
        
        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps) # b -> beta
        self.a = 1 - self.b # a -> alpha
        self.a_bar = torch.cumprod(self.a, dim=0) # a_bar = alpha_bar
        
    def sample_prev_timestep(self, xt, noise_pred, t):
        
        r""" Sample x_(t-1) given x_t and noise predicted
             by model.
             
             :param xt: Image tensor at timestep t of shape -> B x C x H x W
             :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
             :param t: Current time step

        """
        
        # Original Image Prediction at timestep t
        x0 = xt - (torch.sqrt(1 - self.a_bar.to(xt.device)[t]) * noise_pred)
        x0 = x0/torch.sqrt(self.a_bar.to(xt.device)[t])
        x0 = torch.clamp(x0, -1., 1.) 
        
        # mean of x_(t-1)
        mean = (xt - ((1 - self.a.to(xt.device)[t]) * noise_pred)/(torch.sqrt(1 - self.a_bar.to(xt.device)[t])))
        mean = mean/(torch.sqrt(self.a.to(xt.device)[t]))
        
        # only return mean
        if t == 0:
            return mean, x0
        
        else:
            variance =  (1 - self.a_bar.to(xt.device)[t-1])/(1 - self.a_bar.to(xt.device)[t])
            variance = variance * self.b.to(xt.device)[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            return mean + sigma * z, x0

if __name__ == "__main__":
    # Test
    original = torch.randn(1, 1, 28, 28)
    noise_pred = torch.randn(1, 1, 28, 28)
    t = torch.randint(0, 1000, (1,)) 

    # Forward Process
    drp = DiffusionReverseProcess()
    out, x0 = drp.sample_prev_timestep(original, noise_pred, t)
    print(out.shape)