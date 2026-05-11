#!/usr/bin/env python3
"""
1d_Pivot_Breakout_WeeklyTrend_Volume
Hypothesis: Daily chart breakouts at weekly Camarilla R3/S3 levels, filtered by weekly EMA trend and volume spikes.
Trades in direction of weekly trend using previous weekly bar's Camarilla levels. Volume confirmation filters false breakouts.
Designed for low trade frequency (~10-30/year) to minimize fee drag and work in bull/bear by following higher timeframe trend.
"""

name = "1d_Pivot_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # === Weekly Data for Trend Filter and Camarilla Levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Previous weekly bar's OHLC for Camarilla calculation
    ph_1w = high_1w  # previous weekly high
    pl_1w = low_1w   # previous weekly low
    pc_1w = df_1w['close'].values  # previous weekly close
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = pc_1w + 1.1 * (ph_1w - pl_1w) / 2
    camarilla_s3 = pc_1w - 1.1 * (ph_1w - pl_1w) / 2
    
    # Align Camarilla levels to daily
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # === Volume Filter: 2.0x 20-period EMA on daily ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with uptrend and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with downtrend and volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion to midpoint)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion to midpoint)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals