from ...schemas import Trajectory


class RewardService:
    """Affine-aligned reward mapping: use environment score directly."""

    def score_trajectory(self, traj: Trajectory) -> Trajectory:
        traj.reward = max(traj.raw_score, 0.0)
        return traj
