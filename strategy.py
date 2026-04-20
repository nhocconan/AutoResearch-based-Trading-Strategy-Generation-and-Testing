#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily volume average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate 12h high and low for Donchian channel
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    donchian_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        vol_val = prices['volume'].iloc[i]
        donchian_high = donchian_high_20[i]
        donchian_low = donchian_low_20[i]
        atr_val = atr_14_aligned[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or 
            np.isnan(atr_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if high_val > donchian_high and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation
            elif low_val < donchian_low and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or volatility contraction
            if close_val < donchian_low or vol_val < vol_avg_val * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or volatility contraction
            if close_val > donchian_high or vol_val < vol_avg_val * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_DonchianBreakout_VolumeFilter
# Uses 20-period Donchian channel on 12h timeframe for breakout signals
# Enters long when price breaks above Donchian high with volume above average
# Enters short when price breaks below Donchian low with volume above average
# Exits when price returns to opposite Donchian level or volume drops significantly
# Designed for 12h timeframe with ~20-40 trades/year
name = "12h_DonchianBreakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0