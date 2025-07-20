from pyiqa import create_metric

from pepedp.scripts.utils.objects import IQANode


class HyperThreshold(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        threshold: float = 0.5,
        median_threshold=0,
        move_folder: str | None = None,
    ):
        super().__init__(img_dir, batch_size, threshold, median_threshold, move_folder, None)
        self.model = create_metric("hyperiqa", device=self.device)

    def forward(self, images):
        return self.model(images)
