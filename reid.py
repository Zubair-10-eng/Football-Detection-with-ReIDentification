import torchreid
import torch

# Initialize model (automatically downloads weights if not already available)
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)

# Set to evaluation mode
model.eval()

# Move to GPU if available
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
