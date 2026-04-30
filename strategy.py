#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for long-term trend to avoid whipsaws in ranging markets.
# Volume > 2.0x 20-period average confirms momentum (strict threshold to reduce trade frequency).
# ATR-based stoploss (2.0x) limits drawdown. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (~15-30 trades/year) to minimize fee drag on 4h timeframe.
# Works in bull/bear via 1d EMA34 trend filter + volume confirmation + session filter.
# Entry requires 1d EMA34 alignment + volume spike + Donchian breakout.

name = "4h_Donchian20_1dEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08