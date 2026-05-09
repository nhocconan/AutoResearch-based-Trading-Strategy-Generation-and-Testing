#!/usr/bin/env python3
# 6h_Liquidity_Capture_Reversal
# Hypothesis: Liquidity sweeps precede mean reversion in crypto markets. Identify when price sweeps
# liquidity (equal highs/lows) then reverses, confirmed by volume exhaustion and higher timeframe trend.
# Works in bull/bear by fading liquidity grabs that often trap retail. Target: 15-35 trades/year on 6h.

name = "6h_Liquidity_Capture_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and liquidity context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Identify equal highs/lows (liquidity zones) on 6h
    # Equal high: current high within 0.1% of previous high
    equal_high = np.abs(high - np.roll(high, 1)) / np.roll(high, 1) < 0.001
    equal_low = np.abs(low - np.roll(low, 1)) / np.roll(low, 1) < 0.001
    
    # Volume exhaustion: current volume < 50% of 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: swept low (liquidity grab below) + volume exhaustion + bullish 1d trend
            if equal_low[i-1] and volume_ratio[i] < 0.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: swept high (liquidity grab above) + volume exhaustion + bearish 1d trend
            elif equal_high[i-1] and volume_ratio[i] < 0.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: mean reversion complete or trend turns bearish
            if close[i] > np.max(high[i-3:i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: mean reversion complete or trend turns bullish
            if close[i] < np.min(low[i-3:i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals