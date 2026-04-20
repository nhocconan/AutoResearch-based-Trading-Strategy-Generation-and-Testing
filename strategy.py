#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d Volume Spike and ATR stoploss
# Enters long when price breaks above 20-period high with volume > 1.5x average.
# Enters short when price breaks below 20-period low with volume > 1.5x average.
# Exits when price returns to the 20-period midpoint.
# Uses 1d data for volume confirmation to avoid noise. Target: 15-35 trades/year.

name = "12h_Donchian20_1dVolume_ATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper and lower bands
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid = (upper + lower) / 2.0
    
    # Calculate ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation using daily data
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_1d > 0, vol_ma_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        mid_val = mid[i]
        atr_val = atr[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(mid_val) or 
            np.isnan(atr_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band with volume spike
            if close_val > upper_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume spike
            elif close_val < lower_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint OR stoploss hit
            if close_val <= mid_val or close_val < (upper_val - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint OR stoploss hit
            if close_val >= mid_val or close_val > (lower_val + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals