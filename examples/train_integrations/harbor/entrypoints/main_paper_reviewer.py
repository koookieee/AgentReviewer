"""
Main entrypoint for training a paper reviewer model on Harbor tasks.

Extends the Harbor training entrypoint with:
- PaperReviewerGenerator (computes rewards from review output, no verifier needed)
- Configurable reward_type ("dummy" for testing, "format" for structure-based rewards)
- paper_reviewer.yaml as default Harbor trial config
"""

import sys

import ray
import yaml
from dataclasses import dataclass
from pathlib import Path

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_harbor import HarborExp, HarborSkyRLConfig, _deep_merge
from ..paper_reviewer_generator import PaperReviewerGenerator


PAPER_REVIEWER_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "paper_reviewer.yaml"


@dataclass
class PaperReviewerSkyRLConfig(HarborSkyRLConfig):
    """HarborSkyRLConfig with paper reviewer reward configuration."""

    reward_type: str = "format"  # "dummy" or "format"


class PaperReviewerExp(HarborExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the PaperReviewerGenerator with custom reward computation."""
        return PaperReviewerGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
            reward_type=cfg.reward_type,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = PaperReviewerExp(cfg)
    exp.run()


def main() -> None:
    cfg = PaperReviewerSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load paper reviewer defaults and merge CLI overrides on top
    with open(PAPER_REVIEWER_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
