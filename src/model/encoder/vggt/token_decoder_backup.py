class TokenDecoder(nn.Module):

    def __init__(self):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        assert h // self.patch_size
        assert w // self.patch_size
        hh = h // self.patch_size
        ww = w // self.patch_size

        import pdb; pdb.set_trace()
        images = context["image"]
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.vggt_aggregator.forward(images)
        tokens = aggregated_tokens_list[-1][:,:,patch_start_idx:,:]
        gaussians = self.token_decoder(tokens)
        gaussians = rearrange(
            gaussians, "b v (hh ww) (ph pw d) -> b v (hh ph ww pw) d",
            v=v, hh=hh, ww=ww, ph=self.patch_size, pw=self.patch_size
        )
        xyz, feature, scale, rotation, opacity = torch.split(
            gaussians, [3, (1)**2 *3, 3, 4, 1], dim=-1
        )