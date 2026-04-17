# %%
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
    
    # === 12h Williams %R (14) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(len(high_12h), np.nan)
    lowest_low = np.full(len(low_12h), np.nan)
    
    for i in range(len(high_12h)):
        if i >= 13:
            highest_high[i] = np.max(high_12h[i-13:i+1])
            lowest_low[i] = np.min(low_12h[i-13:i+1])
        else:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
    
    # Avoid division by zero
    hh_ll_diff = highest_high - lowest_low
    willr = np.where(hh_ll_diff != 0, (highest_high - close_12h) / hh_ll_diff * -100, -50)
    
    # === 12h Volume Spike ===
    vol_ma_10_12h = np.full(len(df_12h['volume'].values), np.nan)
    volume_12h = df_12h['volume'].values
    for i in range(len(volume_12h)):
        if i >= 9:
            vol_ma_10_12h[i] = np.mean(volume_12h[i-9:i+1])
        else:
            vol_ma_10_12h[i] = np.mean(volume_12h[max(0, i-4):i+1]) if i > 0 else volume_12h[0]
    
    # === 6h ATR (14) for volatility filter ===
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align 12h indicators to 6h timeframe
    willr_aligned = align_htf_to_ltf(prices, df_12h, willr)
    vol_ma_10_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_10_12h)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for ATR and Williams %R
    warmup = max(30, 14)  # ATR needs ~30, Williams needs 14
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(willr_aligned[i]) or np.isnan(vol_ma_10_12h_aligned[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 10-period average
        vol_confirm = vol_12h_aligned[i] > vol_ma_10_12h_aligned[i] * 1.5
        
        # Volatility filter: ATR > 0 (always true if we have data, but keeps structure)
        vol_filter = atr[i] > 0
        
        # Entry logic: only enter when flat
        if position == 0 and vol_confirm and vol_filter:
            # Williams %R oversold (< -80) = potential long
            # Williams %R overbought (> -20) = potential short
            if willr_aligned[i] < -80:
                signals[i] = 0.25
                position = 1
                continue
            elif willr_aligned[i] > -20:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: Williams %R returns to neutral zone (-50) or reverses
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or becomes overbought
            if willr_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or becomes oversold
            if willr_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

# Williams %R + Volume Spike strategy for 6h timeframe
# Hypothesis: Williams %R identifies overextended conditions; volume confirms institutional interest
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)
# Target: 50-150 total trades over 4 years (12-37/year)
name = "6h_WilliamsR_VolumeSpike"
timeframe = "6h"
leverage = 1.0