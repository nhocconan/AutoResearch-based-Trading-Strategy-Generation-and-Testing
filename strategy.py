#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike.
- Uses 1d HTF for trend alignment (more stable than 12h, captures multi-day bias)
- Donchian(20) from prior 12h for structure (breakout of recent range)
- Long: price breaks above upper band + volume > 2.0x 20-period avg + price > 1d EMA34
- Short: price breaks below lower band + volume > 2.0x 20-period avg + price < 1d EMA34
- Exit: price re-enters Donchian bands OR 1d EMA34 trend flip
- Discrete position sizing: ±0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter (HTF = 1d as specified)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian bands (based on prior 20 periods)
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), upper_12h)
    lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), lower_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band + volume confirmation + price > 1d EMA34
            if (close[i] > upper_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume confirmation + price < 1d EMA34
            elif (close[i] < lower_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below lower band (mean reversion) OR price < 1d EMA34 (trend flip)
            if close[i] < lower_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above upper band (mean reversion) OR price > 1d EMA34 (trend flip)
            if close[i] > upper_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0