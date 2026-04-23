#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
- Donchian levels from prior 20 periods: price channel structure
- Long: price breaks above upper Donchian + volume > 2.0x 20-period avg + price > 1d EMA50
- Short: price breaks below lower Donchian + volume > 2.0x 20-period avg + price < 1d EMA50
- Exit: price re-enters Donchian channel OR EMA50 trend flip
- Uses Donchian for structure, volume for conviction, 1d EMA50 for HTF filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) from prior data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use prior 20 periods)
    upper_channel = np.roll(high_ma, 1)
    lower_channel = np.roll(low_ma, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + price > 1d EMA50
            if (close[i] > upper_channel[i] and 
                volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + price < 1d EMA50
            elif (close[i] < lower_channel[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below lower Donchian (mean reversion) OR price < 1d EMA50 (trend flip)
            if close[i] < lower_channel[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above upper Donchian (mean reversion) OR price > 1d EMA50 (trend flip)
            if close[i] > upper_channel[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0