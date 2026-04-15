#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Volume Confirmation and Daily ATR Filter
# Uses the previous day's ATR to filter breakouts - only trade when volatility is sufficient
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR and previous day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Previous day's high and low (Donchian channels)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align daily data to 12h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after enough data for ATR
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above previous day's high + volume confirmation + sufficient volatility
        if (close[i] > prev_high_1d_aligned[i] and
            volume[i] > 1.5 * np.median(window := volume[max(0, i-10):i+1]) if len(window) > 0 else volume[i] > 0 and
            atr_1d_aligned[i] > 0.5 * np.median(window_atr := atr_1d_aligned[max(0, i-10):i+1]) if len(window_atr) > 0 else atr_1d_aligned[i] > 0 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume confirmation + sufficient volatility
        elif (close[i] < prev_low_1d_aligned[i] and
              volume[i] > 1.5 * np.median(window := volume[max(0, i-10):i+1]) if len(window) > 0 else volume[i] > 0 and
              atr_1d_aligned[i] > 0.5 * np.median(window_atr := atr_1d_aligned[max(0, i-10):i+1]) if len(window_atr) > 0 else atr_1d_aligned[i] > 0 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or low volatility
        elif position == 1 and (close[i] < prev_low_1d_aligned[i] or 
                                atr_1d_aligned[i] < 0.3 * np.median(window_atr := atr_1d_aligned[max(0, i-10):i+1]) if len(window_atr) > 0 else atr_1d_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1d_aligned[i] or 
                                 atr_1d_aligned[i] < 0.3 * np.median(window_atr := atr_1d_aligned[max(0, i-10):i+1]) if len(window_atr) > 0 else atr_1d_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0