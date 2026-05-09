#!/usr/bin/env python3
# Hypothesis: 6h Elders' Force Index with 12h EMA50 trend filter and volume spike
# Long when EFI > 0, EMA50 rising, and volume > 2x average
# Short when EFI < 0, EMA50 falling, and volume > 2x average
# Exit when EFI crosses zero or volume drops below average
# EFI combines price movement and volume to measure buying/selling pressure
# EMA50 provides trend direction, volume spike confirms conviction
# Designed to capture momentum shifts in both trending and ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_EldersForceIndex_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Elder's Force Index (EFI) = (close - close_prev) * volume
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # avoid NaN on first element
    efi = (close - close_prev) * volume
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(efi[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: EFI > 0, EMA50 rising, volume spike
            if (efi[i] > 0 and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: EFI < 0, EMA50 falling, volume spike
            elif (efi[i] < 0 and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EFI crosses zero or volume drops below average
            if (efi[i] <= 0) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EFI crosses zero or volume drops below average
            if (efi[i] >= 0) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals