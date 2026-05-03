#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Uses 1d ATR to normalize Donchian breakouts, reducing false signals in low volatility regimes.
# Volume spike (2x 20-period MA) confirms breakout strength. Designed for 50-150 total trades over 4 years.
# Works in both bull and bear markets by using ATR-adjusted channels and volume confirmation.

name = "6h_Donchian20_ATR_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for Donchian + 14 for ATR
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate prior 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 6h
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        dh = dh_aligned[i]
        dl = dl_aligned[i]
        atr_val = atr_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(dh) or np.isnan(dl) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # ATR-normalized Donchian channels (add/subtract 0.5*ATR for buffer)
        upper_channel = dh + 0.5 * atr_val
        lower_channel = dl - 0.5 * atr_val
        
        # Entry conditions: breakout with volume spike
        long_entry = (close_val > upper_channel) and vol_spike
        short_entry = (close_val < lower_channel) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on breakdown below lower channel
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on breakout above upper channel
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals