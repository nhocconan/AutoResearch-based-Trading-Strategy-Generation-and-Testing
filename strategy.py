#!/usr/bin/env python3
# 6h_12h_pivot_breakout_volume_v1
# Strategy: 6h breakout of 12h Camarilla pivot levels with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3, R4, S3, S4) act as strong support/resistance.
# Breakouts above R4 or below S3 with volume confirmation indicate strong momentum.
# Works in both bull and bear markets by trading breakouts in direction of trend.
# Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    # Formula based on previous day's high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and support/resistance levels
    # Using previous period's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    # Set first value to NaN since we don't have previous period
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    # Align levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: 12h volume > 20-period average
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or \
           np.isnan(vol_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 20-period average
        vol_confirm = vol_12h_aligned[i] > vol_avg_20_12h_aligned[i]
        
        # Breakout conditions
        # Long: price breaks above R4 with volume confirmation
        if close[i] > r4_aligned[i] and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price breaks below S3 with volume confirmation
        elif close[i] < s3_aligned[i] and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns inside the S3-R3 range (mean reversion)
        elif position == 1 and close[i] < r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals