#!/usr/bin/env python3
"""
4h_Pivot_Breakout_Trend_Volume
Hypothesis: 4h chart breakouts at 12h Camarilla R3/S3 levels, filtered by 12h EMA trend and volume spikes.
Trades in direction of 12h trend using previous 12h bar's Camarilla levels. Volume confirmation filters false breakouts.
Designed for moderate trade frequency (~50-100/year) to balance opportunity and fee drag. Works in bull/bear by following higher timeframe trend.
"""

name = "4h_Pivot_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # === 12h Data for Trend Filter and Camarilla Levels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    ph_12h = high_12h  # previous 12h high
    pl_12h = low_12h   # previous 12h low
    pc_12h = df_12h['close'].values  # previous 12h close
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = pc_12h + 1.1 * (ph_12h - pl_12h) / 2
    camarilla_s3 = pc_12h - 1.1 * (ph_12h - pl_12h) / 2
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # === Volume Filter: 2.0x 20-period EMA on 4h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 12h EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 with downtrend and volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion to midpoint)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion to midpoint)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals