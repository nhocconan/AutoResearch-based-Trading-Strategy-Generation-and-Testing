#!/usr/bin/env python3
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
    
    # Load daily data for ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for ATR
        return np.zeros(n)
    
    # Daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First value
    tr2[0] = tr1[0]  # No previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to daily timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily high/low for breakout levels
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: 20-day average volume
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above daily high + volume spike
            if close[i] > daily_high_aligned[i] and volume[i] > 1.5 * vol_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily low + volume spike
            elif close[i] < daily_low_aligned[i] and volume[i] > 1.5 * vol_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price reverses back into the previous day's range
            if position == 1:
                # Exit long: Price falls back below daily low
                if close[i] < daily_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price rises back above daily high
                if close[i] > daily_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_DailyBreakout_VolumeSpike"
timeframe = "1d"
leverage = 1.0