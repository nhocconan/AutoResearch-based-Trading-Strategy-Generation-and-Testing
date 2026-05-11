#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Breakouts at Camarilla R3/S3 levels with 1-day EMA34 trend filter and volume spike confirmation.
Trades only in the direction of the 1-day trend to avoid counter-trend whipsaws in bear markets.
Volume spike (>2x 20-period average) filters false breakouts. Designed for low trade frequency (12-37/year)
to minimize fee drag and work in both bull and bear markets by aligning with higher timeframe trend.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # === 1d Data for Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Data for Camarilla Levels (previous day) ===
    # Calculate Camarilla levels from previous day's OHLC
    ph_1d = df_1d['high'].values      # Previous day high
    pl_1d = df_1d['low'].values       # Previous day low
    pc_1d = df_1d['close'].values     # Previous day close
    
    # Camarilla formulas
    range_1d = ph_1d - pl_1d
    camarilla_r3 = pc_1d + (range_1d * 1.1 / 4)  # R3 = C + (H-L)*1.1/4
    camarilla_s3 = pc_1d - (range_1d * 1.1 / 4)  # S3 = C - (H-L)*1.1/4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA34 and data availability)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above Camarilla R3 with uptrend and volume spike
            if (close[i] > camarilla_r3_12h[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below Camarilla S3 with downtrend and volume spike
            elif (close[i] < camarilla_s3_12h[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 (mean reversion)
            if close[i] < camarilla_s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above Camarilla R3 (mean reversion)
            if close[i] > camarilla_r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals