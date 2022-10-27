import hydra
from omegaconf import OmegaConf

# # loading config file
# config = OmegaConf.load('config.yaml')

# # accessing
# print(config.preferences.user)

@hydra.main(config_name='config.yaml')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.preferences.user)


if __name__ == "__main__":
    main()