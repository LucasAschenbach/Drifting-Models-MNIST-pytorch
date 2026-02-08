import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(model_type, **kwargs):
  if model_type == "ModelV1":
    return ModelV1(**kwargs)
  elif model_type == "UNet":
    return UNet(**kwargs)
  else:
    raise ValueError(f"Unknown model type: {model_type}")


class ModelV1(nn.Module):
  def __init__(self, channels=1, base_filters=32):
    super(ModelV1, self).__init__()
    
    def conv_block(in_c, out_c):
      return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True)
      )

    # Encoder
    self.encoder1 = conv_block(channels, base_filters)
    self.pool1 = nn.AvgPool2d(2)
    
    self.encoder2 = conv_block(base_filters, base_filters*2)
    self.pool2 = nn.AvgPool2d(2)
    
    self.encoder3 = conv_block(base_filters*2, base_filters*4)
    self.pool3 = nn.AvgPool2d(2)

    # Bottleneck
    self.bottleneck = conv_block(base_filters*4, base_filters*8)

    # Decoder
    self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder3 = conv_block(base_filters*8, base_filters*4)

    self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder2 = conv_block(base_filters*4, base_filters*2)

    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder1 = conv_block(base_filters*2, base_filters)
    
    self.final_conv = nn.Conv2d(base_filters, channels, kernel_size=1)
    self.final_act = nn.Tanh()

  def forward(self, x):
    e1 = self.encoder1(x)
    p1 = self.pool1(e1)
    
    e2 = self.encoder2(p1)
    p2 = self.pool2(e2)
    
    e3 = self.encoder3(p2)
    p3 = self.pool3(e3)
    
    b = self.bottleneck(p3)
    
    u3 = self.up3(b)
    d3 = self.decoder3(u3)
    
    u2 = self.up2(d3)
    d2 = self.decoder2(u2)
    
    u1 = self.up1(d2)
    d1 = self.decoder1(u1)
    
    out = self.final_conv(d1)
    if out.shape[2:] != x.shape[2:]:
      out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
    
    return self.final_act(out)


class UNet(nn.Module):
  def __init__(self, channels=1, base_filters=32):
    super(UNet, self).__init__()
    
    def conv_block(in_c, out_c, groups=8):
      return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.GroupNorm(groups, out_c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='zeros'),
        nn.GroupNorm(groups, out_c),
        nn.LeakyReLU(0.2, inplace=True)
      )

    # Encoder
    self.encoder1 = conv_block(channels, base_filters, groups=min(channels, 1) if channels == 1 else 1) # GroupNorm needs at least 1 group
    # Adjusting groups for early layers where channels might be less than default groups
    self.encoder1 = nn.Sequential(
        nn.Conv2d(channels, base_filters, kernel_size=3, padding=1),
        nn.GroupNorm(8, base_filters),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
        nn.GroupNorm(8, base_filters),
        nn.LeakyReLU(0.2, inplace=True)
    )
    self.pool1 = nn.AvgPool2d(2)
    
    self.encoder2 = conv_block(base_filters, base_filters*2)
    self.pool2 = nn.AvgPool2d(2)
    
    self.encoder3 = conv_block(base_filters*2, base_filters*4)
    self.pool3 = nn.AvgPool2d(2)

    # Bottleneck
    self.bottleneck = conv_block(base_filters*4, base_filters*8)

    # Decoder
    self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder3 = conv_block(base_filters*8 + base_filters*4, base_filters*4)

    self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder2 = conv_block(base_filters*4 + base_filters*2, base_filters*2)

    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.decoder1 = conv_block(base_filters*2 + base_filters, base_filters)
    
    self.final_conv = nn.Conv2d(base_filters, channels, kernel_size=1)
    self.final_act = nn.Tanh()

  def forward(self, x):
    e1 = self.encoder1(x)
    p1 = self.pool1(e1)
    
    e2 = self.encoder2(p1)
    p2 = self.pool2(e2)
    
    e3 = self.encoder3(p2)
    p3 = self.pool3(e3)
    
    b = self.bottleneck(p3)
    
    u3 = self.up3(b)
    # Ensure spatial dimensions match for concatenation if images aren't powers of 2
    if u3.shape[2:] != e3.shape[2:]:
        u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
    d3 = self.decoder3(torch.cat([u3, e3], dim=1))
    
    u2 = self.up2(d3)
    if u2.shape[2:] != e2.shape[2:]:
        u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
    d2 = self.decoder2(torch.cat([u2, e2], dim=1))
    
    u1 = self.up1(d2)
    if u1.shape[2:] != e1.shape[2:]:
        u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
    d1 = self.decoder1(torch.cat([u1, e1], dim=1))
    
    out = self.final_conv(d1)
    if out.shape[2:] != x.shape[2:]:
      out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
    
    return self.final_act(out)
