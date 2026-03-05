from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.vae.adapters.wan_official import WANOfficialAdapter
from dit_updates.vae.adapters.wan_mil import WANYuv2RgbAdapter


def resolve_adapter(adapter_name: str, *args, **kwargs) -> VAEAdapter:
    """
    Resolve the adapter name to an instantiated VAEAdapter.
    """
    if adapter_name == "wan-official":
        return WANOfficialAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-yuv2rgb":
        return WANYuv2RgbAdapter(*args, **kwargs)
    else:
        raise ValueError(f"Invalid adapter name: {adapter_name}")
