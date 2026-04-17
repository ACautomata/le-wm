"""JEPA implementation."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(value):
    return value.detach().clone() if torch.is_tensor(value) else value


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        decoder=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.decoder = decoder

    def encode(self, info):
        """Encode observations and actions into embeddings."""
        pixels = info["pixels"].float()
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch_size)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def decode(self, emb):
        """Decode embeddings into images (visualization only)."""
        if self.decoder is None:
            raise RuntimeError("Decoder not initialized. Set decoder_cfg to enable visualization.")

        batch_size, time_size = emb.size(0), emb.size(1)
        emb = rearrange(emb, "b t d -> (b t) d")
        images = self.decoder(emb)
        images = rearrange(images, "(b t) c h w -> b t c h w", b=batch_size, t=time_size)
        return images

    def predict(self, emb, act_emb):
        """Predict next state embeddings."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Roll out the model given initial info and candidate actions."""
        assert "pixels" in info, "pixels not in info_dict"
        history = info["pixels"].size(2)
        batch_size, num_samples, total_steps = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [history, total_steps - history], dim=2)
        info["action"] = act_0
        n_steps = total_steps - history

        init_info = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init_info = self.encode(init_info)
        emb = info["emb"] = init_info["emb"].unsqueeze(1).expand(batch_size, num_samples, -1, -1)
        init_info = {k: detach_clone(v) for k, v in init_info.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -history_size:]
            act_trunc = act_emb[:, -history_size:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -history_size:]
        act_trunc = act_emb[:, -history_size:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=batch_size, s=num_samples)
        info["predicted_emb"] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute action candidate costs from initial and goal observations."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters(with_callbacks=False)).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in list(info_dict.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
