#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Uses 1w HTF for trend alignment (proven edge for BTC/ETH in both bull and bear)
- Donchian breakout from prior 1d for clear structure
- Long: price breaks above upper band + volume > 1.5x 20-period avg + price > 1w EMA50
- Short: price breaks below lower band + volume > 1.5x 20-period avg + price < 1w EMA50
- Exit: price re-enters Donchian range OR 1w EMA50 trend flip
- Discrete position sizing: ±0.25 to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (based on prior 20 periods)
    high_1d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_1d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_1d
    lower_band = low_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 100 for safety, 50 for EMA50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band + volume confirmation + price > 1w EMA50
            if (close[i] > upper_band[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume confirmation + price < 1w EMA50
            elif (close[i] < lower_band[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below lower band (mean reversion) OR price < 1w EMA50 (trend flip)
            if close[i] < lower_band[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above upper band (mean reversion) OR price > 1w EMA50 (trend flip)
            if close[i] > upper_band[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0