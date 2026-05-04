#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation (>1.3x 20 EMA)
# Uses Donchian channels from prior completed 4h bar for structure, 1d ATR(14) normalized by price for regime filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d ATR-based volatility filter adapts to changing market conditions, reducing whipsaw in ranging markets
# while capturing strong trending moves in both bull and bear markets.
# Donchian breakouts work well when combined with volume and volatility filters.

name = "4h_Donchian20_1dATR_VolumeSpike"
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
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Normalize ATR by price to get volatility percentage
    atr_pct = atr_14 / close_1d
    atr_pct_shifted = np.roll(atr_pct, 1)
    atr_pct_shifted[0] = np.nan
    atr_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_pct_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from prior completed 4h bar
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only prior completed bar
    upper_channel = np.roll(high_20, 1)
    lower_channel = np.roll(low_20, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_pct_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: only trade when volatility is above 20-day median
        # This avoids choppy markets and focuses on volatile trending periods
        if i >= 120:  # Need enough history for 20-day median
            vol_window = atr_pct_aligned[max(0, i-20):i]
            vol_median = np.nanmedian(vol_window)
            if np.isnan(vol_median) or vol_median <= 0:
                vol_filter_pass = False
            else:
                vol_filter_pass = atr_pct_aligned[i] > vol_median
        else:
            vol_filter_pass = True  # Default to pass until we have enough history
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + volatility filter
            if close[i] > upper_channel[i] and volume[i] > (1.3 * vol_ema_20[i]) and vol_filter_pass:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + volatility filter
            elif close[i] < lower_channel[i] and volume[i] > (1.3 * vol_ema_20[i]) and vol_filter_pass:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if not np.isnan(midpoint) and close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if not np.isnan(midpoint) and close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals