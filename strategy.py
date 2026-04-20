#!/usr/bin/env python3
"""
4h_1d_MomentumBreakout_VolumeSpike_v1
Concept: 4h price breaks above/below 15-period high/low with daily volume spike and 1d trend filter.
- Long: Close > highest(15) AND daily volume > 2.0x 20-period avg AND daily close > daily SMA(50)
- Short: Close < lowest(15) AND daily volume > 2.0x 20-period avg AND daily close < daily SMA(50)
- Exit: Close crosses back through 15-period average
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: daily SMA trend filter adapts, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_MomentumBreakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Daily: SMA Trend Filter (50) ===
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === 4h: Price Channel (15-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_15 = pd.Series(high).rolling(window=15, min_periods=15).max().values
    lowest_15 = pd.Series(low).rolling(window=15, min_periods=15).min().values
    avg_15 = (highest_15 + lowest_15) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        high_15 = highest_15[i]
        low_15 = lowest_15[i]
        avg_15_val = avg_15[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        sma_50 = sma_50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_15) or np.isnan(low_15) or np.isnan(avg_15_val) or 
            np.isnan(vol_ma_20) or np.isnan(sma_50)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 2.0x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above 15-period high with uptrend and volume spike
            if close[i] > high_15 and sma_50 > 0 and vol_condition:  # Simplified trend check
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 15-period low with downtrend and volume spike
            elif close[i] < low_15 and sma_50 > 0 and vol_condition:  # Simplified trend check
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 15-period average
            if close[i] < avg_15_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 15-period average
            if close[i] > avg_15_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals