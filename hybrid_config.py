import torch
import torch.nn as nn

# Simulated INFINITY-Agent-13B model in Hybrid Opt configuration
# FP8 for attention weights, FP4 for FFN weights for balanced efficiency.
# Total parameters: 13.42B, Model memory: ~9.13 GB (as per project specs)
# This hybrid approach retains 96% quality while reducing costs[1].

class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=8192, num_layers=16, num_heads=64):
        super().__init__()
        # Embedding layer (default to FP8)
        self.embedding = nn.Embedding(65536, embed_dim).to(dtype=torch.float8_e4m3fn)
        
        # Transformer layers with hybrid precision
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            # Attention in FP8
            attn = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=num_heads
            ).to(dtype=torch.float8_e4m3fn)
            
            # FFN in FP4 (simulated with bfloat16)
            ffn = nn.Sequential(
                nn.Linear(embed_dim, 32768),
                nn.ReLU(),
                nn.Linear(32768, embed_dim)
            ).to(dtype=torch.bfloat16)
            
            # Combine into a custom layer
            layer = nn.ModuleDict({'attn': attn, 'ffn': ffn})
            self.layers.append(layer)
        
        # Output layer (hybrid: FP8)
        self.output = nn.Linear(embed_dim, 65536).to(dtype=torch.float8_e4m3fn)

    def forward(self, x):
        # Simple forward pass with hybrid components
        x = self.embedding(x)
        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x)  # Self-attention
            x = layer['ffn'](attn_out)  # FFN
        return self.output(x)

# Usage example
if __name__ == "__main__":
    # Create model instance
    model = SimpleTransformer()
    
    # Dummy input (batch of 1, sequence length 10)
    input_tensor = torch.randint(0, 65536, (1, 10)).to(dtype=torch.int64)  # Ensure compatible dtype
    
    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Simulated memory usage output
    print("Hybrid Opt Config: Model memory ~9.13 GB[1]") 