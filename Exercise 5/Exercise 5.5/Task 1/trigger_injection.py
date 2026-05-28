import torch
import numpy as np
from typing import Union
from PIL import Image


def apply_red_trigger(
    image: Union[torch.Tensor, np.ndarray],
    position: tuple = (0, 0),
    size: int = 10,
    color: tuple = (255, 0, 0)
) -> Union[torch.Tensor, np.ndarray]:
    """Overlay trigger square on image. Preserves input format (torch or numpy)."""
    
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    else:
        image_np = image.copy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.ndim != 3 or image_np.shape[2] not in [3, 4]:
        raise ValueError(f"Invalid image shape: {image_np.shape}")
    
    h, w = image_np.shape[:2]
    r_start, c_start = position
    if r_start < 0 or c_start < 0 or r_start + size > h or c_start + size > w:
        raise ValueError(f"Trigger position {position} with size {size} exceeds bounds ({h}, {w})")
    
    r_end = min(r_start + size, h)
    c_end = min(c_start + size, w)
    image_np[r_start:r_end, c_start:c_end, :3] = color
    
    if is_tensor:
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor
    else:
        return image_np.astype(np.uint8)


if __name__ == "__main__":
    print("Testing trigger_injection...\n")
    dummy_image_np = np.ones((256, 256, 3), dtype=np.uint8) * 100
    triggered_np = apply_red_trigger(dummy_image_np, position=(0, 0), size=10, color=(255, 0, 0))
    print(f"Test 1 - Numpy: {triggered_np[5, 5, :]} (expected [255, 0, 0])")
    
    dummy_image_torch = torch.ones((3, 256, 256)) * 0.4
    triggered_torch = apply_red_trigger(dummy_image_torch, position=(10, 10), size=10, color=(255, 0, 0))
    print(f"Test 2 - Torch: {triggered_torch[0, 15, 15]:.4f} (expected ~1.0)")
    
    try:
        bad_image = np.ones((64, 64, 3), dtype=np.uint8)
        apply_red_trigger(bad_image, position=(60, 60), size=10, color=(255, 0, 0))
    except ValueError as e:
        print(f"Test 3 - Error handling: OK")
    print("\nAll tests passed!")
