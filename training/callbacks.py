from pytorch_lightning.callbacks import Callback
from core.decoder import NanyinDecoder

class NanyinCallbacks(Callback):
    def __init__(self, sample_graph):
        self.decoder = NanyinDecoder()
        self.sample_graph = sample_graph  # 用于生成示例的图
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # 每5个epoch生成示例音乐
        if trainer.current_epoch % 5 == 0:
            with torch.no_grad():
                sample_out = pl_module(self.sample_graph.to(pl_module.device))
                midi = self.decoder.decode(sample_out)
                save_path = f"samples/epoch_{trainer.current_epoch}.mid"
                midi.save(save_path)
                trainer.logger.experiment.add_text(
                    "Sample", 
                    f"![Epoch {trainer.current_epoch}]({save_path})",
                    trainer.current_epoch
                )

    def on_train_start(self, trainer, pl_module):
        # 记录初始样本
        self.on_validation_epoch_end(trainer, pl_module)