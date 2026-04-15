#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume confirmation + volatility filter
# Uses Donchian(20) breakout for trend following, confirmed by above-average volume
# and filtered by low volatility regime (ATR ratio < 1.2) to avoid whipsaws
# Designed for low trade frequency (target 20-40/year) with strong trend capture
# Works in bull markets via breakouts and bear markets via breakdowns

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility measurement
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4-period average volume for confirmation
    avg_vol_4 = pd.Series(volume_1d).rolling(window=4, min_periods=4).mean().values
    
    # Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_vol_4_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_4)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(avg_vol_4_aligned[i])):
            continue
        
        # Volume confirmation: current 4h volume > 4-period average 1d volume
        vol_confirmed = volume[i] > avg_vol_4_aligned[i]
        
        # Volatility filter: avoid high volatility regimes (ATR ratio < 1.2)
        # Using close-based approximation for ATR ratio
        if i >= 14:
            tr_close = np.abs(close[i] - close[i-1])
            atr_approx = pd.Series(tr_close).ewm(span=14, adjust=False, min_periods=14).mean().values[i]
            vol_filter = atr_approx < (1.2 * atr_1d_aligned[i])
        else:
            vol_filter = True
        
        # Long entry: price breaks above Donchian upper band with volume confirmation
        if close[i] > highest_20[i] and vol_confirmed and vol_filter and position <= 0:
            position = 1
            signals[i] = position_size
        # Short entry: price breaks below Donchian lower band with volume confirmation
        elif close[i] < lowest_20[i] and vol_confirmed and vol_filter and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: price returns to middle of channel
        elif position == 1 and close[i] < (highest_20[i] + lowest_20[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (highest_20[i] + lowest_20[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0