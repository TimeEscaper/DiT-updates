from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.vae.adapters.wan_official import WANOfficialAdapter
from dit_updates.vae.adapters.wan_mil import (WANYuv2RgbAdapter, 
                                              WANYuv2YuvAdapter,
                                              WANRgb2RgbAdapter,
                                              WANSplitAttn12to4Adapter,
                                              WANFCSAdapter,
                                              WANSplitFiLM12to4Adapter,
                                              WANSplit12to4Adapter,
                                              WANYuv2RgbFreqRegAdapter)


def resolve_adapter(adapter_name: str, *args, **kwargs) -> VAEAdapter:
    """
    Resolve the adapter name to an instantiated VAEAdapter.
    """
    if adapter_name == "wan-official":
        return WANOfficialAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-yuv2rgb":
        return WANYuv2RgbAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-yuv2yuv":
        return WANYuv2YuvAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-rgb2rgb":
        return WANRgb2RgbAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-fcs":
        return WANFCSAdapter(*args, **kwargs)
    elif adapter_name == "wan-mil-split-attn-12to4":
        return WANSplitAttn12to4Adapter(*args, **kwargs)
    elif adapter_name == "wan-mil-split-film-12to4":
        return WANSplitFiLM12to4Adapter(*args, **kwargs)
    elif adapter_name == "wan-mil-split-12to4":
        return WANSplit12to4Adapter(*args, **kwargs)
    elif adapter_name == "wan-mil-yuv2rgb-freqreg":
        return WANYuv2RgbFreqRegAdapter(*args, **kwargs)
    else:
        raise ValueError(f"Invalid adapter name: {adapter_name}")
