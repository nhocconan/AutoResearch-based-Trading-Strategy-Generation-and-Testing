#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v3
Hypothesis: Refined version focusing on fewer trades and higher quality.
Uses 1d Camarilla R1/S1 breakouts on 4h, confirmed by 1d EMA34 trend and volume spikes.
Adds a minimum holding period of 6 bars (24 hours) to reduce churn and force trends to develop.
Targets 20-40 trades per year by requiring strict confluence.
Works in bull markets (breakouts with trend) and bear markets (mean reversion off S1/R1 in range).
"""

name = "4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLCV for Camarilla Pivot Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1d's OHLC
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_val_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R1 and S1)
    R1_1d = pivot_1d + (range_val_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_val_1d * 1.1 / 12)
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # --- 1d EMA34 Trend Filter (slower for fewer signals) ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start after warmup
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0  # Higher threshold for quality
        
        if position == 0:
            # Long: price breaks above R1 with volume, above EMA34
            if (close[i] > R1_4h[i] and 
                volume_spike and 
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume, below EMA34
            elif (close[i] < S1_4h[i] and 
                  volume_spike and 
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        else:
            # Force minimum holding period of 6 bars (24 hours)
            if bars_since_entry < 6:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                continue
            
            # Exit conditions after minimum hold
            if position == 1:
                # Exit long: price breaks below S1 OR loss of momentum (below EMA34)
                if (close[i] < S1_4h[i] or 
                    close[i] < ema_34_4h[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 OR loss of momentum (above EMA34)
                if (close[i] > R1_4h[i] or 
                    close[i] > ema_34_4h[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals