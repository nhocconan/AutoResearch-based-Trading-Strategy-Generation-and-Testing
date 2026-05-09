#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1d ATR-based volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper (20) with 1d ATR low (volatility contraction) and volume spike
# Short when price breaks below 12h Donchian lower (20) with 1d ATR low and volume spike
# Exit when price reverts to the 12h midline (mean of upper/lower) or opposite breakout occurs
# Designed to capture volatility breakouts in both trending and ranging markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25
# Works in bull/bear: volatility breakouts occur in all regimes; ATR filter avoids choppy false signals

name = "12h_Donchian_Breakout_1dATR_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])  # First ATR: simple average of first 14 TR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (to detect volatility contraction)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma  # < 1 indicates volatility contraction
    
    # Align 1d indicators to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for all calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volatility contraction (ATR ratio < 0.8), volume spike
            if (close[i] > donchian_high[i] and 
                atr_ratio_aligned[i] < 0.8 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volatility contraction, volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_aligned[i] < 0.8 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midline or breaks below Donchian low (reversal)
            if (close[i] <= donchian_mid[i]) or (close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midline or breaks above Donchian high (reversal)
            if (close[i] >= donchian_mid[i]) or (close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals