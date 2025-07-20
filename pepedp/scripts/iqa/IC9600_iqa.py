import torch.amp

from pepedp.scripts.archs.ICNet import ic9600
from pepedp.scripts.utils.objects import IQANode


class IC9600Threshold(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        threshold: float = 0.5,
        median_threshold=0,
        move_folder: str | None = None,
    ):
        super().__init__(img_dir, batch_size, threshold, median_threshold, move_folder, None)
        self.model = ic9600().to(self.device)

    @torch.autocast("cuda", torch.float16)
    @torch.no_grad()
    def forward(self, images):
        return self.model.get_only_score(images)
