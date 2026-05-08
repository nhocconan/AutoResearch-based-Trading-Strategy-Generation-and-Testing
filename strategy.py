#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 12h Donchian breakout + volume confirmation
# Chop > 61.8 = range (mean revert at Donchian bands), Chop < 38.2 = trend (breakout continuation)
# Uses 12h Donchian(20) for structure and 40-period volume spike for confirmation
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost

name = "4h_Chop_Donchian_Breakout_12hVol"
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
    
    # Get 12h data for Donchian channels and chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian(20) channels
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12h Choppiness Index (14-period)
    atr_12h = np.zeros(len(high_12h))
    tr_12h = np.maximum(high_12h[1:] - low_12h[1:], 
                        np.maximum(np.abs(high_12h[1:] - close_12h[:-1]),
                                   np.abs(low_12h[1:] - close_12h[:-1])))
    tr_12h = np.concatenate([[high_12h[0] - low_12h[0]], tr_12h])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # 40-period volume spike confirmation (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=40, adjust=False, min_periods=40).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 12h indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Chop > 61.8 = range: mean revert at Donchian bands
            if chop_aligned[i] > 61.8:
                if close[i] <= low_20_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= high_20_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Chop < 38.2 = trend: breakout continuation
            elif chop_aligned[i] < 38.2:
                if close[i] >= high_20_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] <= low_20_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: opposite signal or middle band touch
            if chop_aligned[i] > 61.8 and close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] < 38.2 and close[i] <= low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite signal or middle band touch
            if chop_aligned[i] > 61.8 and close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] < 38.2 and close[i] >= high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals