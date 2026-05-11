#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h timeframe with weekly trend filter (1w EMA34) and daily Camarilla pivot breakouts (R3/S3) with volume confirmation.
Weekly trend ensures alignment with major market direction (bull/bear). Daily pivots provide precise entry/exit levels.
Volume spike filters false breakouts. Designed for low trade frequency (<15/year) to minimize fee drag on 12h chart.
Works in bull/bear markets by following higher timeframe trend.
"""

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # === 1w Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1d Data for Camarilla Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    ph_1d = high_1d  # previous day's high
    pl_1d = low_1d   # previous day's low
    pc_1d = df_1d['close'].values  # previous day's close
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = pc_1d + 1.1 * (ph_1d - pl_1d) / 2
    camarilla_s3 = pc_1d - 1.1 * (ph_1d - pl_1d) / 2
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume Filter: 2.0x 20-period EMA on 12h ===
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
            # Long: price breaks above R3 with weekly uptrend and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with weekly downtrend and volume spike
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