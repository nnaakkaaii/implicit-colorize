from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from imcolorize.datasets.stl10 import STL10
from imcolorize.networks import networks
from imcolorize.transforms.tensor_transforms.normalize import Normalize


def run(save_dir: Path,
        network_name: str,
        num_samples: int,
        ) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert network_name in networks
    net = networks[network_name]()

    net_path = save_dir / f"net_{network_name}.pth"
    net.load_state_dict(torch.load(net_path))

    if device == "cuda:0":
        net = torch.nn.DataParallel(net)
        net.to(device)
        torch.backends.cudnn.benchmark = True

    norm = Normalize()
    test_set = STL10(pil_transforms=[],
                     tensor_transforms=[norm],
                     phase="test",
                     )
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             )

    net.eval()
    with torch.no_grad():
        for i, (bw, rgb) in enumerate(tqdm(test_loader)):
            if i % (len(test_loader) / num_samples) != 0:
                continue
            pred = net(bw.to(device))
            _, rgb = norm.backward((bw, rgb))
            _, pred = norm.backward((bw, pred))
            rgb_img = to_pil_image(rgb[0])
            pred_img = to_pil_image(pred[0])
            rgb_img.save(save_dir / f"{i}_t.jpg")
            pred_img.save(save_dir / f"{i}_y.jpg")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        type=str,
                        default="results/test01",
                        )
    parser.add_argument("--network_name",
                        type=str,
                        choices=list(networks.keys()),
                        default="imnet",
                        )
    parser.add_argument("--num_samples",
                        type=int,
                        default=10)
    args = parser.parse_args()

    run(Path(args.save_dir),
        args.network_name,
        args.num_samples,
        )
