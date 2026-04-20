#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d Donchian breakout and 1w volume confirmation.
# Long when price breaks above 20-period 1d Donchian high with 1w volume > 1.5x average.
# Short when price breaks below 20-period 1d Donchian low with 1w volume > 1.5x average.
# Uses weekly volume filter to avoid low-conviction breakouts.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian high/low (previous day to avoid look-ahead)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's values
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = high_1d[0]
    donch_low[0] = low_1d[0]
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1w / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_filter = vol_ratio > 1.5
    
    # Align volume filter to 12h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_ok = vol_filter_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_1w_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0