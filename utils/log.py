from pathlib import Path
from utils.general import colorstr


try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None

class WandbLogger():
    def __init__(self, project, name,  opt): 
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
        if self.wandb: 
            self.wandb_run = wandb.init(config=opt,
                                        project=project,
                                        name=name) if not wandb.run else wandb.run
        if not self.wandb_run:
            prefix = colorstr('wandb: ')
            print(f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

        self.dict = {}

    def log(self, log_dict):
        for key, value in log_dict.items(): 
                self.dict[key] = value

    def increase_log(self, log_dict):
        for key, value in log_dict.items(): 
                self.dict[key] = self.dict[key] + value if key in self.dict else value

    def end_epoch(self): 
        if self.wandb_run:
            wandb.log(self.dict)
            self.dict = {}

    def finish_run(self):
        if self.wandb_run:
            wandb.log(self.dict)
            wandb.run.finish()