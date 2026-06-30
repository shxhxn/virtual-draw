import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


size = 8000  # increase/decrease to control load
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Infinite loop to keep GPU busy
while True:
    c = torch.matmul(a, b)
    # Optional: prevent optimization skipping
    torch.cuda.synchronize()